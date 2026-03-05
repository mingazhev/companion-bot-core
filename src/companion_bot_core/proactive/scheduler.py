"""Background scheduler for proactive check-in messages.

Runs as an ``asyncio.Task`` (created in ``main.py``).  Every
``POLL_INTERVAL_SECONDS`` it queries the Redis sorted set for due
check-ins and sends messages via the Telegram bot.
"""

from __future__ import annotations

import asyncio
import uuid as _uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from companion_bot_core.db.engine import get_async_session
from companion_bot_core.db.models import User, UserProfile
from companion_bot_core.i18n import normalize_locale, tr
from companion_bot_core.logging_config import get_logger
from companion_bot_core.proactive.checkin import (
    get_due_checkins,
    is_in_quiet_hours,
    mark_sent,
    reschedule_tomorrow,
    unschedule_checkin,
)

if TYPE_CHECKING:
    from aiogram import Bot
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine

log = get_logger(__name__)

POLL_INTERVAL_SECONDS = 60


def _parse_timezone(tz_str: str | None) -> timezone:
    """Parse a timezone string like 'UTC+3' or 'UTC-5' into a timezone.

    Returns UTC for unknown/invalid formats.
    """
    if not tz_str:
        return UTC
    tz_str = tz_str.strip().upper()
    if tz_str == "UTC" or tz_str == "GMT":
        return UTC
    for prefix in ("UTC", "GMT"):
        if tz_str.startswith(prefix):
            offset_str = tz_str[len(prefix):]
            try:
                hours = int(offset_str)
                return timezone(timedelta(hours=hours))
            except (ValueError, TypeError):
                return UTC
    return UTC


async def _send_checkin(
    bot: Bot,
    telegram_user_id: int,
    locale: str | None,
) -> bool:
    """Send a check-in message to the user. Returns True on success."""
    resolved = normalize_locale(locale)
    text = tr("checkin.message", resolved)
    try:
        await bot.send_message(chat_id=telegram_user_id, text=text)
        return True
    except Exception:  # noqa: BLE001
        log.warning(
            "checkin_send_failed",
            telegram_user_id=telegram_user_id,
        )
        return False


async def run_checkin_scheduler(
    bot: Bot,
    redis: Redis,
    engine: AsyncEngine,
) -> None:
    """Poll Redis for due check-ins and send messages.

    This function runs indefinitely and should be wrapped in
    ``asyncio.create_task()``.
    """
    log.info("checkin_scheduler_started")

    while True:
        try:
            due_user_ids = await get_due_checkins(redis)
            for user_id_str in due_user_ids:
                try:
                    await _process_one_checkin(
                        bot, redis, engine, user_id_str,
                    )
                except Exception:  # noqa: BLE001
                    log.exception(
                        "checkin_process_error", user_id=user_id_str,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("checkin_scheduler_poll_error")

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _process_one_checkin(
    bot: Bot,
    redis: Redis,
    engine: AsyncEngine,
    user_id_str: str,
) -> None:
    """Process a single due check-in: look up user prefs, check quiet hours, send."""
    user_uuid = _uuid.UUID(user_id_str)

    async with get_async_session(engine) as session:
        # Load user and profile
        q = (
            select(User, UserProfile)
            .outerjoin(UserProfile, User.id == UserProfile.user_id)
            .where(User.id == user_uuid)
        )
        result = await session.execute(q)
        row = result.one_or_none()

        if row is None:
            # User deleted — remove from schedule
            await unschedule_checkin(redis, user_id_str)
            return

        user, profile = row.tuple()

        if profile is None or not profile.proactive_enabled or profile.checkin_time is None:
            # Preferences changed — remove from schedule
            await unschedule_checkin(redis, user_id_str)
            return

        user_tz = _parse_timezone(user.timezone)
        now_local = datetime.now(tz=user_tz).time()

        # Check quiet hours
        if is_in_quiet_hours(now_local, profile.quiet_hours_start, profile.quiet_hours_end):
            log.info("checkin_skipped_quiet_hours", user_id=user_id_str)
            await reschedule_tomorrow(
                redis, user_id_str, profile.checkin_time, user_tz,
            )
            return

        telegram_user_id = user.telegram_user_id
        locale = user.locale

    # Send outside the DB session
    sent = await _send_checkin(bot, telegram_user_id, locale)
    if sent:
        await mark_sent(redis, user_id_str)
        log.info("checkin_sent", user_id=user_id_str)

    # Reschedule for tomorrow regardless of success/failure
    # (re-read checkin_time since we're outside the session)
    async with get_async_session(engine) as session:
        q2 = select(UserProfile.checkin_time).where(UserProfile.user_id == user_uuid)
        result2 = await session.execute(q2)
        checkin_time = result2.scalar_one_or_none()

    if checkin_time is not None:
        await reschedule_tomorrow(redis, user_id_str, checkin_time, user_tz)
