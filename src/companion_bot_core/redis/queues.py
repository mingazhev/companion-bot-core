"""Redis-backed job queues for refinement and retry workflows.

Two FIFO list-based queues are defined:
- ``QUEUE_REFINEMENT_JOBS``: primary queue consumed by the refinement worker.
- ``QUEUE_RETRY_JOBS``: secondary queue for failed jobs awaiting retry.

Producers call ``enqueue_*`` functions; the worker calls ``dequeue_job``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis.asyncio import Redis

QUEUE_REFINEMENT_JOBS = "refinement_jobs"
QUEUE_RETRY_JOBS = "retry_jobs"


async def enqueue_refinement_job(
    redis: Redis,
    user_id: str,
    payload: dict[str, Any],
) -> int:
    """Push a refinement job to the tail of the refinement queue.

    *user_id* is always written into the job envelope, overriding any
    ``user_id`` key already present in *payload*.

    Returns the new queue length after the push.
    """
    job: dict[str, Any] = {**payload, "user_id": user_id}
    result: int = await redis.rpush(QUEUE_REFINEMENT_JOBS, json.dumps(job))  # type: ignore[misc]
    return result


async def enqueue_retry_job(
    redis: Redis,
    user_id: str,
    payload: dict[str, Any],
) -> int:
    """Push a retry job to the tail of the retry queue.

    *user_id* is always written into the job envelope, overriding any
    ``user_id`` key already present in *payload*.

    Returns the new queue length after the push.
    """
    job: dict[str, Any] = {**payload, "user_id": user_id}
    result: int = await redis.rpush(QUEUE_RETRY_JOBS, json.dumps(job))  # type: ignore[misc]
    return result


async def dequeue_job(
    redis: Redis,
    queue: str,
    timeout: int = 0,
) -> dict[str, Any] | None:
    """Pop a job from the head of ``queue`` using a blocking left-pop.

    Args:
        queue: One of ``QUEUE_REFINEMENT_JOBS`` or ``QUEUE_RETRY_JOBS``.
        timeout: Seconds to block waiting for an item; 0 blocks indefinitely.

    Returns:
        Decoded job dict, or ``None`` if the operation timed out.
    """
    result: tuple[str, str] | None = await redis.blpop([queue], timeout=timeout)  # type: ignore[misc]
    if result is None:
        return None
    _queue_name, raw = result
    return json.loads(raw)  # type: ignore[no-any-return]


async def get_queue_length(redis: Redis, queue: str) -> int:
    """Return the number of pending items in ``queue``."""
    length: int = await redis.llen(queue)  # type: ignore[misc]
    return length
