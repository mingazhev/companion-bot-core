"""Bot application factory.

Provides :func:`build_bot` and :func:`build_dispatcher` which wire together
the aiogram :class:`~aiogram.Bot`, :class:`~aiogram.Dispatcher`,
:class:`~tdbot.bot.middleware.IngressMiddleware`, and the command
:data:`~tdbot.bot.handlers.router`.

Polling vs webhook mode is selected by the caller based on
``settings.telegram_webhook_host``:

- Empty string (default) → long-polling via ``dp.start_polling(bot)``.
- Non-empty string → caller is responsible for mounting the webhook endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from tdbot.bot.handlers import router
from tdbot.bot.middleware import IngressMiddleware

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine

    from tdbot.config import Settings
    from tdbot.inference.client import ChatAPIClient
    from tdbot.prompt.snapshot_store import SnapshotStore


def build_bot(settings: Settings) -> Bot:
    """Instantiate the aiogram :class:`~aiogram.Bot` from *settings*.

    The default ``parse_mode`` is set to HTML so handlers can use basic
    HTML formatting without repeating the argument.
    """
    return Bot(
        token=settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


def build_dispatcher(
    settings: Settings,
    engine: AsyncEngine,
    redis: Redis[str],
    snapshot_store: SnapshotStore | None = None,
    chat_client: ChatAPIClient | None = None,
) -> Dispatcher:
    """Create and configure the aiogram :class:`~aiogram.Dispatcher`.

    Registers:
    - :class:`~tdbot.bot.middleware.IngressMiddleware` as an *outer* middleware
      on every incoming :class:`~aiogram.types.Update`.
    - The command :data:`~tdbot.bot.handlers.router`.

    Args:
        settings:       Application settings (token, rate limits, etc.).
        engine:         Async SQLAlchemy engine for per-request DB sessions.
        redis:          Async Redis client for idempotency and rate limiting.
        snapshot_store: Prompt snapshot store injected into message handlers.
        chat_client:    Inference API client injected into message handlers.

    Returns:
        A fully configured :class:`~aiogram.Dispatcher` ready for polling or
        webhook mode.
    """
    dp = Dispatcher()
    middleware = IngressMiddleware(settings=settings, engine=engine, redis=redis)
    dp.update.outer_middleware(middleware)
    dp.include_router(router)
    # Expose shared resources to all handlers via workflow_data
    if snapshot_store is not None:
        dp["snapshot_store"] = snapshot_store
    if chat_client is not None:
        dp["chat_client"] = chat_client
    dp["settings"] = settings
    return dp
