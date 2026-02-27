"""Unit tests for tdbot.inference.circuit_breaker."""

from __future__ import annotations

import asyncio
import time

import pytest

from tdbot.inference.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ok() -> str:
    return "ok"


async def _fail() -> str:
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_state_is_closed() -> None:
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# CLOSED → counts failures
# ---------------------------------------------------------------------------


async def test_success_resets_failure_count() -> None:
    cb = CircuitBreaker(failure_threshold=3)
    # Accumulate two failures
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(_fail)
    assert cb.failure_count == 2

    # A success resets the count
    result = await cb.call(_ok)
    assert result == "ok"
    assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# CLOSED → OPEN after failure_threshold failures
# ---------------------------------------------------------------------------


async def test_circuit_opens_after_threshold_failures() -> None:
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(_fail)

    assert cb.state == CircuitState.OPEN


async def test_open_circuit_raises_circuit_breaker_open() -> None:
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(_fail)

    with pytest.raises(CircuitBreakerOpen) as exc_info:
        await cb.call(_ok)

    assert exc_info.value.failure_count == 2
    assert exc_info.value.reset_at > time.monotonic()


async def test_open_circuit_does_not_call_func() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)

    call_count = 0

    async def _probe() -> None:
        nonlocal call_count
        call_count += 1

    with pytest.raises(CircuitBreakerOpen):
        await cb.call(_probe)

    assert call_count == 0


# ---------------------------------------------------------------------------
# OPEN → HALF_OPEN after recovery_timeout
# ---------------------------------------------------------------------------


async def test_transitions_to_half_open_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)

    assert cb.state == CircuitState.OPEN

    # Simulate time passing beyond recovery_timeout by patching monotonic
    original = time.monotonic()
    monkeypatch.setattr(
        "tdbot.inference.circuit_breaker.time.monotonic",
        lambda: original + 2.0,
    )

    # Next call should be allowed (HALF_OPEN probe)
    result = await cb.call(_ok)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED  # type: ignore[comparison-overlap]


async def test_half_open_failure_returns_to_open(monkeypatch: pytest.MonkeyPatch) -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)

    original = time.monotonic()
    monkeypatch.setattr(
        "tdbot.inference.circuit_breaker.time.monotonic",
        lambda: original + 2.0,
    )

    # Probe fails → back to OPEN
    with pytest.raises(RuntimeError):
        await cb.call(_fail)

    assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# HALF_OPEN → CLOSED after success_threshold successes
# ---------------------------------------------------------------------------


async def test_half_open_requires_success_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, success_threshold=2)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)

    original = time.monotonic()
    monkeypatch.setattr(
        "tdbot.inference.circuit_breaker.time.monotonic",
        lambda: original + 2.0,
    )

    # First success → still HALF_OPEN
    await cb.call(_ok)
    assert cb.state == CircuitState.HALF_OPEN

    # Second success → CLOSED
    await cb.call(_ok)
    assert cb.state == CircuitState.CLOSED  # type: ignore[comparison-overlap]


# ---------------------------------------------------------------------------
# CircuitBreakerOpen message
# ---------------------------------------------------------------------------


def test_circuit_breaker_open_str_contains_info() -> None:
    exc = CircuitBreakerOpen(failure_count=5, reset_at=time.monotonic() + 30.0)
    msg = str(exc)
    assert "5" in msg
    assert "resets_in" in msg


# ---------------------------------------------------------------------------
# Concurrent safety
# ---------------------------------------------------------------------------


async def test_half_open_cancellation_clears_probe_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """asyncio.CancelledError during a HALF_OPEN probe must clear the in-flight
    flag so the circuit breaker does not get stuck permanently."""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)

    assert cb.state == CircuitState.OPEN

    original = time.monotonic()
    t = original + 2.0
    monkeypatch.setattr(
        "tdbot.inference.circuit_breaker.time.monotonic",
        lambda: t,
    )

    async def _cancelled() -> str:
        raise asyncio.CancelledError

    # Probe is cancelled — must not leave the breaker stuck.
    # The cancellation counts as a failure, returning the circuit to OPEN.
    with pytest.raises(asyncio.CancelledError):
        await cb.call(_cancelled)

    assert cb.state == CircuitState.OPEN

    # Advance time past recovery_timeout again so the circuit allows a new probe.
    t = original + 4.0
    monkeypatch.setattr(
        "tdbot.inference.circuit_breaker.time.monotonic",
        lambda: t,
    )

    # A new probe should be accepted (flag was cleared by the cancellation path)
    result = await cb.call(_ok)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED  # type: ignore[comparison-overlap]


async def test_half_open_cancel_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """CancelledError between a successful probe return and _record_success must
    still clear the in-flight flag so the circuit does not get permanently stuck."""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)

    assert cb.state == CircuitState.OPEN

    original = time.monotonic()
    t = original + 2.0
    monkeypatch.setattr(
        "tdbot.inference.circuit_breaker.time.monotonic",
        lambda: t,
    )

    # Patch _record_success to simulate CancelledError during success recording
    # (e.g. task cancelled while awaiting the lock inside _record_success).
    original_record_success = cb._record_success

    async def _cancel_on_success() -> None:
        raise asyncio.CancelledError

    cb._record_success = _cancel_on_success  # type: ignore[assignment]

    async def _succeeds() -> str:
        return "ok"

    with pytest.raises(asyncio.CancelledError):
        await cb.call(_succeeds)

    # The probe flag must be cleared even though CancelledError interrupted
    # _record_success.
    assert cb._half_open_probe_in_flight is False

    # Restore and advance time for the next probe.
    cb._record_success = original_record_success  # type: ignore[assignment]
    t = original + 4.0
    monkeypatch.setattr(
        "tdbot.inference.circuit_breaker.time.monotonic",
        lambda: t,
    )

    result = await cb.call(_ok)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED  # type: ignore[comparison-overlap]


async def test_concurrent_failures_open_circuit() -> None:
    """Multiple concurrent failing calls should eventually open the circuit.

    Once the failure threshold is reached, remaining concurrent tasks receive
    CircuitBreakerOpen instead of the underlying RuntimeError.
    Both exceptions are acceptable here; the key assertion is that the circuit
    ends up OPEN after all tasks settle.
    """
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

    async def _failing_task() -> None:
        with pytest.raises((RuntimeError, CircuitBreakerOpen)):
            await cb.call(_fail)

    await asyncio.gather(*[_failing_task() for _ in range(5)])
    assert cb.state == CircuitState.OPEN
