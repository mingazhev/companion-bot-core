"""Daily check-in helpers for proactive messaging.

Manages Redis-backed scheduling of daily check-in messages.  Users opt in
via ``/checkin on HH:MM`` and out via ``/checkin off``.

Redis key layout
----------------
``checkin:schedule`` — sorted set: ``{user_uuid}`` → next fire Unix timestamp.
``checkin:last:{user_uuid}`` — timestamp of the last check-in sent (7-day TTL).
"""

from __future__ import annotations

import time as _time
from datetime import UTC, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

from companion_bot_core.logging_config import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

log = get_logger(__name__)

_SCHEDULE_KEY = "checkin:schedule"
_LAST_SENT_PREFIX = "checkin:last"
_LAST_SENT_TTL = 7 * 86400  # 7 days


def is_in_quiet_hours(
    now_time: time,
    quiet_start: time | None,
    quiet_end: time | None,
) -> bool:
    """Return True if *now_time* falls within the quiet window.

    Handles wrapping across midnight (e.g., 22:00 — 08:00).
    """
    if quiet_start is None or quiet_end is None:
        return False

    if quiet_start <= quiet_end:
        # Same-day range (e.g. 08:00 — 12:00)
        return quiet_start <= now_time <= quiet_end

    # Wraps midnight (e.g. 22:00 — 08:00)
    return now_time >= quiet_start or now_time <= quiet_end


def compute_next_fire(
    checkin_time: time,
    user_tz: timezone | None = None,
) -> float:
    """Return the next fire timestamp (UTC epoch) for the given check-in time.

    If the check-in time has already passed today (in the user's timezone),
    schedules for tomorrow.
    """
    tz = user_tz or UTC
    now = datetime.now(tz=tz)
    candidate = datetime.combine(now.date(), checkin_time, tzinfo=tz)

    if candidate <= now:
        candidate += timedelta(days=1)

    return candidate.timestamp()


async def schedule_checkin(
    redis: Redis,
    user_id: str,
    checkin_time: time,
    user_tz: timezone | None = None,
) -> float:
    """Add or update the user's check-in in the Redis sorted set.

    Returns the next fire timestamp.
    """
    next_fire = compute_next_fire(checkin_time, user_tz)
    await redis.zadd(_SCHEDULE_KEY, {user_id: next_fire})
    log.info("checkin_scheduled", user_id=user_id, next_fire=next_fire)
    return next_fire


async def unschedule_checkin(redis: Redis, user_id: str) -> None:
    """Remove the user from the check-in schedule."""
    await redis.zrem(_SCHEDULE_KEY, user_id)
    log.info("checkin_unscheduled", user_id=user_id)


async def get_due_checkins(redis: Redis) -> list[str]:
    """Return user IDs whose check-in time has arrived.

    Fetches all members with score <= now from the sorted set.
    """
    now = _time.time()
    members = await redis.zrangebyscore(_SCHEDULE_KEY, "-inf", str(now))
    return [m.decode() if isinstance(m, bytes) else m for m in members]


async def mark_sent(redis: Redis, user_id: str) -> None:
    """Record that a check-in was sent and reschedule for tomorrow."""
    key = f"{_LAST_SENT_PREFIX}:{user_id}"
    await redis.set(key, str(int(_time.time())), ex=_LAST_SENT_TTL)


async def reschedule_tomorrow(
    redis: Redis,
    user_id: str,
    checkin_time: time,
    user_tz: timezone | None = None,
) -> None:
    """Reschedule the user's check-in for the next day."""
    tz = user_tz or UTC
    now = datetime.now(tz=tz)
    tomorrow = datetime.combine(
        now.date() + timedelta(days=1), checkin_time, tzinfo=tz,
    )
    await redis.zadd(_SCHEDULE_KEY, {user_id: tomorrow.timestamp()})
