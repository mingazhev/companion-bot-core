"""aiogram outer middleware: idempotency, rate limiting, and user provisioning.

Each incoming Telegram Update is processed through three gates:

1. **Idempotency** — duplicate ``update_id`` values (Telegram retry behaviour)
   are silently dropped using a Redis SET NX EX key.
2. **Global rate limit** — the per-instance request-per-second cap guards
   against floods that would exhaust downstream resources.
3. **Per-user rate limit** — prevents individual users from overwhelming the
   bot; exceeded requests are silently dropped.

When all gates pass the middleware provisions the internal :class:`~tdbot.db.models.User`
row (creating it on first contact) and injects ``db_user``, ``db_session``,
and ``tg_user`` into the handler data dict so downstream handlers can depend
on them via aiogram's magic injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiogram import BaseMiddleware

from tdbot.bot.users import get_or_create_user
from tdbot.db.engine import get_async_session
from tdbot.logging_config import get_logger
from tdbot.redis.idempotency import mark_update_seen
from tdbot.redis.rate_limit import check_global_rate_limit, check_user_rate_limit

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiogram.types import Update
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine

    from tdbot.config import Settings

log = get_logger(__name__)


class IngressMiddleware(BaseMiddleware):
    """Outer middleware applied to every :class:`~aiogram.types.Update`.

    Args:
        settings: Application settings (rate limit values, etc.).
        engine: Async SQLAlchemy engine used to open per-request sessions.
        redis: Async Redis client for idempotency keys and rate limit counters.
    """

    def __init__(self, settings: Settings, engine: AsyncEngine, redis: Redis[str]) -> None:
        self._settings = settings
        self._engine = engine
        self._redis = redis

    async def __call__(  # type: ignore[override]
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        update_id = event.update_id

        # ------------------------------------------------------------------ #
        # Gate 1 — Idempotency: drop updates we have already processed.
        # ------------------------------------------------------------------ #
        is_new = await mark_update_seen(self._redis, update_id)
        if not is_new:
            log.info("duplicate_update_dropped", update_id=update_id)
            return None

        # ------------------------------------------------------------------ #
        # Gate 2 — Global rate limit.
        # ------------------------------------------------------------------ #
        within_global = await check_global_rate_limit(
            self._redis,
            max_rps=self._settings.rate_limit_global_rps,
        )
        if not within_global:
            log.warning("global_rate_limit_exceeded", update_id=update_id)
            return None

        # ------------------------------------------------------------------ #
        # Identify the Telegram user who sent this update.
        # ------------------------------------------------------------------ #
        tg_user = None
        if event.message and event.message.from_user:
            tg_user = event.message.from_user
        elif event.callback_query and event.callback_query.from_user:
            tg_user = event.callback_query.from_user

        if tg_user is None:
            # System or channel updates — pass through without user context.
            return await handler(event, data)

        # ------------------------------------------------------------------ #
        # Gate 3 — Per-user rate limit.
        # ------------------------------------------------------------------ #
        within_user = await check_user_rate_limit(
            self._redis,
            user_id=str(tg_user.id),
            max_requests=self._settings.rate_limit_messages_per_minute,
        )
        if not within_user:
            log.warning(
                "user_rate_limit_exceeded",
                telegram_user_id=tg_user.id,
                update_id=update_id,
            )
            return None

        # ------------------------------------------------------------------ #
        # Provision internal User row and inject context into handler data.
        # ------------------------------------------------------------------ #
        async with get_async_session(self._engine) as session:
            db_user = await get_or_create_user(session, tg_user.id)
            data["db_user"] = db_user
            data["db_session"] = session
            data["tg_user"] = tg_user
            log.debug(
                "user_provisioned",
                telegram_user_id=tg_user.id,
                internal_user_id=str(db_user.id),
                update_id=update_id,
            )
            return await handler(event, data)
