"""Async-aware circuit breaker for the model inference client.

States
------
CLOSED    Normal operation; failures are counted against a threshold.
OPEN      Fast-fail mode; all calls raise ``CircuitBreakerOpen`` immediately.
HALF_OPEN One probe is allowed after ``recovery_timeout`` seconds; a
          successful probe resets to CLOSED, a failed probe returns to OPEN.

The implementation is single-process and uses an asyncio.Lock so state
transitions are safe under concurrent coroutines.
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):  # noqa: N818
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, failure_count: int, reset_at: float) -> None:
        self.failure_count = failure_count
        self.reset_at = reset_at
        remaining = max(0.0, reset_at - time.monotonic())
        super().__init__(
            f"Circuit breaker is open (failures={failure_count}, "
            f"resets_in={remaining:.1f}s)"
        )


class CircuitBreaker:
    """Simple async-safe circuit breaker.

    Args:
        failure_threshold: Number of consecutive failures required to open the
            circuit (default 5).
        recovery_timeout: Seconds to wait in OPEN state before allowing a probe
            (default 60.0).
        success_threshold: Consecutive successes in HALF_OPEN required to close
            the circuit again (default 1).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public read-only properties (no lock needed — informational only)
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    # ------------------------------------------------------------------
    # Internal state transitions (all hold _lock)
    # ------------------------------------------------------------------

    async def _record_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_probe_in_flight = False
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._opened_at = None
            elif self._state == CircuitState.CLOSED:
                # Reset rolling failure count on success while closed.
                self._failure_count = 0

    async def _record_failure(self) -> None:
        async with self._lock:
            self._half_open_probe_in_flight = False
            self._failure_count += 1
            self._success_count = 0
            if (
                self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
                and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    async def _check_state(self) -> None:
        """Raise CircuitBreakerOpen if the circuit is open.

        Also transitions OPEN → HALF_OPEN when ``recovery_timeout`` has elapsed.
        Only one probe is allowed in HALF_OPEN; concurrent callers are rejected
        until the probe completes (preventing a thundering-herd on recovery).
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0.0)
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    self._half_open_probe_in_flight = True
                else:
                    reset_at = (self._opened_at or 0.0) + self.recovery_timeout
                    raise CircuitBreakerOpen(self._failure_count, reset_at)
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_probe_in_flight:
                    reset_at = (self._opened_at or 0.0) + self.recovery_timeout
                    raise CircuitBreakerOpen(self._failure_count, reset_at)
                # Re-arm the gate so only one probe runs at a time.
                # This is essential when success_threshold > 1: after a
                # successful probe that does not yet meet the threshold,
                # _record_success clears the flag; the next _check_state
                # call re-arms it here, ensuring one-at-a-time semantics.
                self._half_open_probe_in_flight = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *func* subject to the circuit breaker policy.

        Args:
            func: Async callable to protect.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func*.

        Raises:
            CircuitBreakerOpen: When the circuit is OPEN and recovery timeout
                has not yet elapsed.
            Exception: Any exception raised by *func* is re-raised after
                recording the failure.
        """
        await self._check_state()
        try:
            result: Any = await func(*args, **kwargs)
        except BaseException:
            await self._record_failure()
            raise
        try:
            await self._record_success()
        except BaseException:
            # If CancelledError (or another BaseException) interrupts
            # success recording, clear the probe flag directly so the
            # circuit breaker does not get permanently stuck in HALF_OPEN.
            self._half_open_probe_in_flight = False
            raise
        return result
