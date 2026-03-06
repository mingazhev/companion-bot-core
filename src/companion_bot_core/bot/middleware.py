"""aiogram outer middleware: idempotency, rate limiting, and user provisioning.

Each incoming Telegram Update is processed through three gates:

1. **Idempotency** — duplicate ``update_id`` values (Telegram retry behaviour)
   are silently dropped using a Redis SET NX EX key.
2. **Global rate limit** — the per-instance request-per-second cap guards
   against floods that would exhaust downstream resources.
3. **Per-user rate limit** — prevents individual users from overwhelming the
   bot; exceeded requests are silently dropped.

When all gates pass, the middleware provisions the internal
:class:`~companion_bot_core.db.models.User` row (creating it on first contact)
and injects ``db_user``, ``db_session``, and ``tg_user`` into the handler data
dict so downstream handlers can depend on them via aiogram's magic injection.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from aiogram import BaseMiddleware
from aiogram.enums import ChatType

from companion_bot_core.bot.users import get_or_create_user
from companion_bot_core.db.engine import get_async_session
from companion_bot_core.i18n import tr
from companion_bot_core.logging_config import bind_correlation_id, get_logger
from companion_bot_core.proactive.checkin import (
    extract_deferred_checkin_ops,
    flush_deferred_checkin_ops,
)
from companion_bot_core.prompt.postgres_store import (
    extract_deferred_lock_releases,
    extract_deferred_redis_writes,
    flush_deferred_lock_releases,
    flush_deferred_redis_writes,
)
from companion_bot_core.redis.idempotency import clear_update_key, mark_update_seen
from companion_bot_core.redis.rate_limit import check_global_rate_limit, check_user_rate_limit

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiogram.types import Update
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine

    from companion_bot_core.config import Settings

log = get_logger(__name__)


class IngressMiddleware(BaseMiddleware):
    """Outer middleware applied to every :class:`~aiogram.types.Update`.

    Args:
        settings: Application settings (rate limit values, etc.).
        engine: Async SQLAlchemy engine used to open per-request sessions.
        redis: Async Redis client for idempotency keys and rate limit counters.
    """

    def __init__(self, settings: Settings, engine: AsyncEngine, redis: Redis) -> None:
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

        # Bind the Telegram update_id as the correlation ID so all log records
        # within this request carry a consistent identifier.
        bind_correlation_id(str(update_id))

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
        max_rpm = self._settings.rate_limit_messages_per_minute
        user_request_count = await check_user_rate_limit(
            self._redis,
            user_id=str(tg_user.id),
            max_requests=max_rpm,
        )
        if user_request_count > max_rpm:
            log.warning(
                "user_rate_limit_exceeded",
                telegram_user_id=tg_user.id,
                update_id=update_id,
            )
            # Notify the user only on the first exceeded request to avoid spam.
            if user_request_count == max_rpm + 1 and event.message:
                try:
                    locale = tg_user.language_code
                    text = tr("rate_limit.exceeded", locale)
                    is_group = event.message.chat.type in (
                        ChatType.GROUP, ChatType.SUPERGROUP,
                    )
                    if is_group:
                        await event.message.reply(text)
                    else:
                        await event.message.answer(text)
                except Exception:  # noqa: BLE001
                    log.debug("rate_limit_notify_failed", update_id=update_id)
            return None

        # ------------------------------------------------------------------ #
        # Provision internal User row and inject context into handler data.
        # ------------------------------------------------------------------ #
        deferred_writes: list[tuple[str, str]] = []
        deferred_locks: list[tuple[str, str]] = []
        deferred_checkin: list[tuple[str, ...]] = []
        try:
            async with get_async_session(self._engine) as session:
                db_user = await get_or_create_user(session, tg_user.id)
                data["db_user"] = db_user
                data["db_session"] = session
                data["tg_user"] = tg_user
                data["redis"] = self._redis
                log.debug(
                    "user_provisioned",
                    telegram_user_id=tg_user.id,
                    internal_user_id=str(db_user.id),
                    update_id=update_id,
                )
                result = await handler(event, data)
                # Extract deferred writes and lock releases while the session
                # is still open (session.info is inaccessible after close).
                deferred_writes = extract_deferred_redis_writes(session)
                deferred_locks = extract_deferred_lock_releases(session)
                deferred_checkin = extract_deferred_checkin_ops(session)
        except Exception:
            # Handler or DB commit failed — clear the idempotency key so
            # Telegram retries of this update_id are not permanently dropped.
            try:
                await clear_update_key(self._redis, update_id)
            except Exception:  # noqa: BLE001
                log.warning(
                    "idempotency_key_cleanup_failed",
                    update_id=update_id,
                )
            # Release any deferred profile locks immediately.  The DB commit
            # failed so no profile state was persisted and there is no stale
            # Redis pointer risk.  Release now so the user can retry without
            # waiting for the 30-second TTL.
            if deferred_locks:
                try:
                    await flush_deferred_lock_releases(deferred_locks, self._redis)
                except Exception:  # noqa: BLE001
                    log.warning(
                        "deferred_lock_release_failed_on_error",
                        update_id=update_id,
                    )
            raise

        # Transaction committed — flush snapshot-pointer Redis writes
        # that were deferred during the handler to avoid referencing
        # uncommitted rows.  If this fails we log but do NOT clear the
        # idempotency key: the handler already completed and the DB
        # committed, so allowing a Telegram retry would duplicate work.
        # Redis SET is idempotent so callers can safely retry.
        redis_flush_ok = False
        for _attempt in range(3):
            try:
                await flush_deferred_redis_writes(deferred_writes, self._redis)
                redis_flush_ok = True
                break
            except Exception:  # noqa: BLE001
                if _attempt == 2:  # noqa: PLR2004
                    log.error(
                        "deferred_redis_flush_failed",
                        update_id=update_id,
                    )
                else:
                    await asyncio.sleep(0.25 * (_attempt + 1))

        # If all flush attempts failed, delete the stale active pointer keys
        # so the next get_active() falls back to the DB and returns the most
        # recently committed snapshot, rather than serving the old pointer
        # indefinitely.  Deleting is safe: get_active() treats a missing key
        # as "no pointer" and queries the DB for the highest-version snapshot
        # (the newly committed one), which is the correct recovery behaviour.
        stale_cleanup_ok = False
        if not redis_flush_ok and deferred_writes:
            try:
                stale_keys = [key for key, _ in deferred_writes]
                await self._redis.delete(*stale_keys)
                stale_cleanup_ok = True
            except Exception:  # noqa: BLE001
                log.warning(
                    "deferred_redis_pointer_cleanup_failed",
                    update_id=update_id,
                )

        # Flush deferred checkin schedule operations (ZADD/ZREM) after
        # the DB transaction has committed so that Redis and DB stay in sync.
        # Retry up to 3 times (matching the snapshot flush pattern) because
        # the user has already received a success message from the handler.
        if deferred_checkin:
            for _checkin_attempt in range(3):
                try:
                    await flush_deferred_checkin_ops(deferred_checkin, self._redis)
                    break
                except Exception:  # noqa: BLE001
                    if _checkin_attempt == 2:  # noqa: PLR2004
                        log.error(
                            "deferred_checkin_flush_failed",
                            update_id=update_id,
                        )
                    else:
                        await asyncio.sleep(0.25 * (_checkin_attempt + 1))

        # Release profile write locks after the Redis active pointer is in a
        # known-good state: either the pointer was written successfully
        # (redis_flush_ok) or the stale pointer was deleted so the next
        # get_active() falls back to the DB (stale_cleanup_ok).  Only hold
        # the lock until TTL expiry when both paths failed and a stale
        # pointer could still exist, to prevent concurrent writers from
        # reading stale Redis state.
        if deferred_locks and (redis_flush_ok or stale_cleanup_ok):
            try:
                await flush_deferred_lock_releases(deferred_locks, self._redis)
            except Exception:  # noqa: BLE001
                log.warning(
                    "deferred_lock_release_failed",
                    update_id=update_id,
                )

        return result
