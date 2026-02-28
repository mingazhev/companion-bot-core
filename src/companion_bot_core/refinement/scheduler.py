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

from companion_bot_core.logging_config import get_logger
from companion_bot_core.redis.queues import enqueue_refinement_job

if TYPE_CHECKING:
    from redis.asyncio import Redis

log = get_logger(__name__)

_LAST_SCHEDULED_KEY_PREFIX = "refinement:last_scheduled"
# Must match the guard key used by _maybe_enqueue_refinement in the orchestrator.
_REFINEMENT_GUARD_PREFIX = "refinement:pending"
_REFINEMENT_GUARD_TTL = 600  # 10 minutes


def _last_scheduled_key(user_id: str) -> str:
    return f"{_LAST_SCHEDULED_KEY_PREFIX}:{user_id}"


async def should_schedule_by_cadence(
    redis: Redis,
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
    redis: Redis,
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
    redis: Redis,
    user_id: str,
    cadence_seconds: int,
) -> bool:
    """Enqueue a refinement job if the cadence interval has elapsed.

    Checks the cadence **and** the shared ``refinement:pending`` guard so
    that a cadence-triggered job is not enqueued when an activity-threshold
    job is already in flight.

    Note: the check-then-act sequence between the cadence read and the
    guard SET-NX is **not** fully atomic; concurrent callers may still
    slip through, but the guard makes duplicates far less likely.

    Args:
        redis:            Async Redis client.
        user_id:          String representation of the user's UUID.
        cadence_seconds:  Minimum seconds between refinement jobs for this user.

    Returns:
        True if a job was enqueued; False if the cadence interval has not yet
        elapsed or a refinement job is already in flight.
    """
    if not await should_schedule_by_cadence(redis, user_id, cadence_seconds):
        return False

    # Acquire the shared guard so activity-threshold and cadence triggers
    # cannot enqueue concurrently.
    guard_key = f"{_REFINEMENT_GUARD_PREFIX}:{user_id}"
    acquired = await redis.set(guard_key, "1", nx=True, ex=_REFINEMENT_GUARD_TTL)
    if not acquired:
        log.debug("refinement_cadence_skipped_guard", user_id=user_id)
        return False

    try:
        await enqueue_refinement_job(redis, user_id, {"trigger": "cadence"})
    except Exception:  # noqa: BLE001
        try:
            await redis.delete(guard_key)
        except Exception:  # noqa: BLE001
            log.warning("refinement_cadence_guard_cleanup_failed", user_id=user_id)
        log.warning("refinement_cadence_enqueue_failed", user_id=user_id)
        return False

    try:
        await record_refinement_scheduled(redis, user_id, cadence_seconds)
    except Exception:  # noqa: BLE001
        # Job is already enqueued; the guard prevents duplicate enqueues until
        # its TTL expires.  Logging here is sufficient — no guard cleanup.
        log.warning("refinement_cadence_record_failed", user_id=user_id)

    log.info("refinement_cadence_job_enqueued", user_id=user_id)
    return True
