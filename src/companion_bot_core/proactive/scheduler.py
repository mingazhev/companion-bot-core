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
from companion_bot_core.orchestrator.habits import get_active_habits
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


def parse_timezone(tz_str: str | None) -> timezone:
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

        user_tz = parse_timezone(user.timezone)
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
        checkin_time = profile.checkin_time

    # Send outside the DB session
    sent = await _send_checkin(bot, telegram_user_id, locale)
    if sent:
        await mark_sent(redis, user_id_str)
        log.info("checkin_sent", user_id=user_id_str)

    # Send habit reminders for unchecked habits (once per day per habit)
    await _send_habit_reminders(bot, redis, engine, user_id_str, telegram_user_id, locale)

    # Reschedule for tomorrow regardless of success/failure
    if checkin_time is not None:
        await reschedule_tomorrow(redis, user_id_str, checkin_time, user_tz)


_HABIT_REMINDER_PREFIX = "habit:reminder"
_HABIT_REMINDER_TTL = 86400  # 1 day


async def _send_habit_reminders(
    bot: Bot,
    redis: Redis,
    engine: AsyncEngine,
    user_id_str: str,
    telegram_user_id: int,
    locale: str | None,
) -> None:
    """Send reminders for habits not yet checked today (once per day per habit)."""
    user_uuid = _uuid.UUID(user_id_str)
    resolved = normalize_locale(locale)
    now = datetime.now(tz=UTC)

    try:
        async with get_async_session(engine) as session:
            habits = await get_active_habits(session, user_uuid)

        for habit in habits:
            # Skip if already checked today
            if habit.last_checked_at is not None and habit.last_checked_at.date() == now.date():
                continue
            # Skip weekly habits that aren't due
            if habit.frequency == "weekly" and habit.last_checked_at is not None:
                days_since = (now - habit.last_checked_at).days
                if days_since < 7:  # noqa: PLR2004
                    continue

            # Dedup: only remind once per day per habit
            guard_key = f"{_HABIT_REMINDER_PREFIX}:{user_id_str}:{habit.id}"
            guard_set = await redis.set(guard_key, "1", nx=True, ex=_HABIT_REMINDER_TTL)
            if not guard_set:
                continue

            text = tr("habit.reminder", resolved, title=habit.title)
            try:
                await bot.send_message(chat_id=telegram_user_id, text=text)
                log.info(
                    "habit_reminder_sent",
                    user_id=user_id_str,
                    habit_title=habit.title,
                )
            except Exception:  # noqa: BLE001
                log.warning(
                    "habit_reminder_send_failed",
                    user_id=user_id_str,
                    habit_title=habit.title,
                )
    except Exception:  # noqa: BLE001
        log.warning("habit_reminders_failed", user_id=user_id_str)
