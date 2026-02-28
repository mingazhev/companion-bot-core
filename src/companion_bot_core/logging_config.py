"""Structured logging with per-request correlation IDs.

Usage
-----
At application startup call `configure_logging(settings)` once.

Anywhere in request-handling code:

    from companion_bot_core.logging_config import bind_correlation_id, get_logger

    bind_correlation_id("some-uuid")          # stores in contextvars
    log = get_logger(__name__)
    log.info("message received", user_id=123)

The correlation ID is automatically injected into every log record produced
in the same async task / thread until it is replaced or cleared.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, cast

import structlog

from companion_bot_core.privacy.pii_redactor import redact_pii

if TYPE_CHECKING:
    from companion_bot_core.config import Settings

# ---------------------------------------------------------------------------
# Correlation-ID context variable
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def new_correlation_id() -> str:
    """Generate and store a fresh correlation ID for the current context."""
    cid = uuid.uuid4().hex
    _correlation_id.set(cid)
    return cid


def bind_correlation_id(cid: str) -> None:
    """Store an externally-provided correlation ID (e.g. Telegram update ID)."""
    _correlation_id.set(cid)


def get_correlation_id() -> str:
    """Return the correlation ID for the current async context (may be empty)."""
    return _correlation_id.get()


# ---------------------------------------------------------------------------
# Span-ID context variable (used by companion_bot_core.tracing)
# ---------------------------------------------------------------------------

_span_id: ContextVar[str] = ContextVar("span_id", default="")


def bind_span_id(sid: str) -> Token[str]:
    """Set the active span ID and return a reset token for later restoration."""
    return _span_id.set(sid)


def reset_span_id(token: Token[str]) -> None:
    """Restore the span ID to its value before :func:`bind_span_id` was called."""
    _span_id.reset(token)


def get_span_id() -> str:
    """Return the current span ID (empty string when not inside a span)."""
    return _span_id.get()


# ---------------------------------------------------------------------------
# structlog processors that inject correlation and span IDs
# ---------------------------------------------------------------------------


def _inject_correlation_id(
    _logger: Any,
    _method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    cid = _correlation_id.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def _inject_span_id(
    _logger: Any,
    _method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    sid = _span_id.get()
    if sid:
        event_dict["span_id"] = sid
    return event_dict


# ---------------------------------------------------------------------------
# Public configuration entry point
# ---------------------------------------------------------------------------


def configure_logging(settings: Settings) -> None:
    """Wire structlog and stdlib logging according to *settings*.

    Call once at application startup before any log statements are issued.
    """
    log_level_name = settings.log_level.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_correlation_id,
        _inject_span_id,
        redact_pii,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Suppress noisy third-party loggers
    for noisy in ("aiogram", "asyncpg", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger with the given module name."""
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
