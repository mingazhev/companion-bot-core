"""Cadence-based refinement job scheduler.

Complements the activity-threshold scheduling in the orchestrator by gating
refinement jobs behind a minimum time interval (cadence) per user.  This
prevents flooding the queue when a user is very active.

Public surface:
    should_schedule_by_cadence(redis, user_id, cadence_seconds)  -> bool
    record_refinement_scheduled(redis, user_id, cadence_seconds) -> None
    enqueue_if_cadence_due(redis, user_id, cadence_seconds)      -> bool
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tdbot.logging_config import get_logger
from tdbot.redis.queues import enqueue_refinement_job

if TYPE_CHECKING:
    from redis.asyncio import Redis

log = get_logger(__name__)

_LAST_SCHEDULED_KEY_PREFIX = "refinement:last_scheduled"


def _last_scheduled_key(user_id: str) -> str:
    return f"{_LAST_SCHEDULED_KEY_PREFIX}:{user_id}"


async def should_schedule_by_cadence(
    redis: Redis[str],
    user_id: str,
    cadence_seconds: int,
) -> bool:
    """Return True if enough time has elapsed since the last scheduled refinement.

    Returns True when no previous refinement has been scheduled (first run).

    Args:
        redis:            Async Redis client.
        user_id:          String representation of the user's UUID.
        cadence_seconds:  Minimum seconds between consecutive refinement jobs.
    """
    raw: str | None = await redis.get(_last_scheduled_key(user_id))
    if raw is None:
        return True
    elapsed = time.time() - float(raw)
    return elapsed >= cadence_seconds


async def record_refinement_scheduled(
    redis: Redis[str],
    user_id: str,
    cadence_seconds: int,
) -> None:
    """Record the current timestamp as the last refinement schedule time.

    The key is given a TTL of ``2 * cadence_seconds`` so that it auto-expires
    when a user becomes inactive for an extended period.

    Args:
        redis:            Async Redis client.
        user_id:          String representation of the user's UUID.
        cadence_seconds:  Used to compute the key TTL.
    """
    await redis.set(
        _last_scheduled_key(user_id),
        str(time.time()),
        ex=cadence_seconds * 2,
    )


async def enqueue_if_cadence_due(
    redis: Redis[str],
    user_id: str,
    cadence_seconds: int,
) -> bool:
    """Enqueue a refinement job if the cadence interval has elapsed.

    Atomically checks the cadence, enqueues the job, and records the timestamp
    so subsequent calls within the interval are skipped.

    Args:
        redis:            Async Redis client.
        user_id:          String representation of the user's UUID.
        cadence_seconds:  Minimum seconds between refinement jobs for this user.

    Returns:
        True if a job was enqueued; False if the cadence interval has not yet
        elapsed.
    """
    if not await should_schedule_by_cadence(redis, user_id, cadence_seconds):
        return False

    await enqueue_refinement_job(redis, user_id, {"trigger": "cadence"})
    await record_refinement_scheduled(redis, user_id, cadence_seconds)
    log.info("refinement_cadence_job_enqueued", user_id=user_id)
    return True
