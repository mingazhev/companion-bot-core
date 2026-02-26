"""Redis-backed per-user abuse throttle.

Tracks policy-violation events per user in a sliding window.  When a user
accumulates too many violations they are temporarily blocked; the block
automatically expires via Redis TTL.

Key layout
----------
- ``abuse:violations:{user_id}`` — sorted set of violation timestamps
- ``abuse:block:{user_id}``     — string key set when user is blocked (TTL-based)

Public surface:
    record_policy_violation  — increment per-user violation counter
    is_user_abuse_blocked    — check whether a user is currently blocked
    get_violation_count      — read current violation count (without incrementing)
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis.asyncio import Redis

_VIOLATIONS_PREFIX = "abuse:violations:"
_BLOCK_PREFIX = "abuse:block:"

# Rolling window for counting violations (seconds).
_VIOLATION_WINDOW_SECONDS = 300  # 5 minutes

# Number of violations within the window before the user is blocked.
_BLOCK_THRESHOLD = 5

# How long (seconds) a block remains active before expiring automatically.
_BLOCK_TTL_SECONDS = 600  # 10 minutes

# Message returned to an abuse-blocked user.
ABUSE_BLOCK_MESSAGE = (
    "You've triggered too many policy violations in a short time. "
    "Please wait a few minutes before sending more messages."
)


def _violation_key(user_id: str) -> str:
    return f"{_VIOLATIONS_PREFIX}{user_id}"


def _block_key(user_id: str) -> str:
    return f"{_BLOCK_PREFIX}{user_id}"


async def is_user_abuse_blocked(redis: Redis, user_id: str) -> bool:
    """Return True if the user is currently in an active abuse-block period.

    Args:
        redis:   Active Redis client (decode_responses=True).
        user_id: Opaque string identifier for the user.

    Returns:
        True if a block key exists for this user; False otherwise.
    """
    result = await redis.exists(_block_key(user_id))
    return bool(result)


async def record_policy_violation(
    redis: Redis,
    user_id: str,
    *,
    violation_window_seconds: int = _VIOLATION_WINDOW_SECONDS,
    block_threshold: int = _BLOCK_THRESHOLD,
    block_ttl_seconds: int = _BLOCK_TTL_SECONDS,
) -> bool:
    """Record a policy violation for *user_id* and apply a block if threshold is reached.

    Each call adds one entry to the per-user sliding-window counter.  If the
    resulting count meets or exceeds *block_threshold*, a block key is set with
    a TTL of *block_ttl_seconds*.

    Args:
        redis:                    Active Redis client.
        user_id:                  Opaque string identifier for the user.
        violation_window_seconds: Rolling window for counting violations.
        block_threshold:          Violations within window before blocking.
        block_ttl_seconds:        TTL (seconds) for the block key.

    Returns:
        True if a new block was applied; False if the user remains unblocked.
    """
    now = time.time()
    window_start = now - violation_window_seconds
    vkey = _violation_key(user_id)
    member = str(uuid.uuid4())

    # Atomic pipeline: prune old entries, add new one, count, refresh TTL.
    pipe = redis.pipeline()
    pipe.zremrangebyscore(vkey, "-inf", window_start)
    pipe.zadd(vkey, {member: now})
    pipe.zcard(vkey)
    pipe.expire(vkey, violation_window_seconds + 1)
    results: list[Any] = await pipe.execute()

    count: int = int(results[2])

    if count >= block_threshold:
        # Set the block key; it expires automatically.
        await redis.set(_block_key(user_id), "1", ex=block_ttl_seconds)
        return True

    return False


async def get_violation_count(
    redis: Redis,
    user_id: str,
    *,
    violation_window_seconds: int = _VIOLATION_WINDOW_SECONDS,
) -> int:
    """Return the number of violations recorded for *user_id* within the window.

    Does not add a new entry.  Useful for monitoring and tests.

    Args:
        redis:                    Active Redis client.
        user_id:                  Opaque string identifier for the user.
        violation_window_seconds: Rolling window for counting violations.

    Returns:
        Number of violation entries within the current window.
    """
    now = time.time()
    window_start = now - violation_window_seconds
    vkey = _violation_key(user_id)

    pipe = redis.pipeline()
    pipe.zremrangebyscore(vkey, "-inf", window_start)
    pipe.zcard(vkey)
    results: list[Any] = await pipe.execute()
    return int(results[1])


async def clear_abuse_block(redis: Redis, user_id: str) -> None:
    """Remove the active abuse block for *user_id* (e.g. for admin intervention).

    Args:
        redis:   Active Redis client.
        user_id: Opaque string identifier for the user.
    """
    await redis.delete(_block_key(user_id))
