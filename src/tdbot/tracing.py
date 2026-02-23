"""Lightweight in-process tracing via contextvars.

Spans are represented as structured log events emitted at span boundaries.
The current span ID is stored in a :class:`~contextvars.ContextVar` and
automatically injected into every structlog record produced within the span
via the processor added in :mod:`tdbot.logging_config`.

Usage
-----
Async spans (for coroutines)::

    from tdbot.tracing import span

    async with span("orchestrator.process_message", user_id=user_id_str):
        reply = await process_message(...)

Synchronous spans (for plain functions)::

    from tdbot.tracing import sync_span

    with sync_span("detector.classify"):
        result = classify(text)

The current span ID can be read at any point with :func:`get_span_id`.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any

from tdbot.logging_config import bind_span_id, get_span_id, reset_span_id

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator
    from contextvars import Token

# Re-export so callers can read the current span ID without importing internals.
__all__ = ["get_span_id", "span", "sync_span"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_span_id() -> str:
    """Generate a random 16-hex-char span identifier."""
    return uuid.uuid4().hex[:16]


def _enter_span(name: str, extra: dict[str, Any]) -> tuple[str, str, float, Token[str]]:
    """Open a new span: update the span-ID contextvar and emit a start event.

    Returns
    -------
    tuple[str, str, float, Token[str]]
        ``(span_id, parent_span_id, start_time, reset_token)``
    """
    from tdbot.logging_config import get_logger  # local import avoids top-level cycle

    log = get_logger("tdbot.tracing")

    parent = get_span_id()
    sid = _new_span_id()
    token = bind_span_id(sid)
    start = time.perf_counter()

    fields: dict[str, Any] = {"span_name": name, "span_id": sid, **extra}
    if parent:
        fields["parent_span_id"] = parent
    log.debug("span_start", **fields)

    return sid, parent, start, token


def _exit_span(
    name: str,
    sid: str,
    parent: str,
    start: float,
    token: Token[str],
    exc: BaseException | None,
) -> None:
    """Close a span: reset the contextvar and emit an end or error event."""
    from tdbot.logging_config import get_logger  # local import avoids top-level cycle

    log = get_logger("tdbot.tracing")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    fields: dict[str, Any] = {"span_name": name, "span_id": sid, "elapsed_ms": elapsed_ms}
    if parent:
        fields["parent_span_id"] = parent

    reset_span_id(token)

    if exc is not None:
        log.error("span_error", error=str(exc), **fields)
    else:
        log.debug("span_end", **fields)


# ---------------------------------------------------------------------------
# Public span context managers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def span(name: str, **kwargs: Any) -> AsyncGenerator[None, None]:
    """Async span context manager.

    Emits ``span_start`` and ``span_end`` (or ``span_error``) log events
    with elapsed time in milliseconds.  The span ID is available inside the
    block via :func:`get_span_id` and is automatically injected into
    structlog records.

    Args:
        name:      Human-readable span name, e.g. ``"ingress.handle_message"``.
        **kwargs:  Additional key-value pairs logged with ``span_start``.
    """
    sid, parent, start, token = _enter_span(name, kwargs)
    captured_exc: BaseException | None = None
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        captured_exc = exc
        raise
    finally:
        _exit_span(name, sid, parent, start, token, captured_exc)


@contextmanager
def sync_span(name: str, **kwargs: Any) -> Generator[None, None, None]:
    """Synchronous span context manager.

    Same semantics as :func:`span` but for synchronous code paths.

    Args:
        name:      Human-readable span name.
        **kwargs:  Additional key-value pairs logged with ``span_start``.
    """
    sid, parent, start, token = _enter_span(name, kwargs)
    captured_exc: BaseException | None = None
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        captured_exc = exc
        raise
    finally:
        _exit_span(name, sid, parent, start, token, captured_exc)
