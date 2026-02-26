"""Redis-backed sliding-window rate limiting.

Two scopes are supported:

- **Per-user**: enforces a message-per-minute cap per Telegram user.
- **Global**: enforces an overall requests-per-second cap across all users.

Algorithm
---------
Each request is recorded in a Redis sorted set with the current wall-clock
timestamp as its score and a UUID as its member (guarantees uniqueness even
under concurrent calls).  Before counting, entries older than the window are
pruned with ZREMRANGEBYSCORE.  All four commands (prune, add, count, refresh
TTL) are executed atomically inside a single pipeline.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis.asyncio import Redis

_PREFIX_USER = "rate_limit:user:"
_KEY_GLOBAL = "rate_limit:global"
_DEFAULT_WINDOW_SECONDS = 60


def _user_key(user_id: str) -> str:
    return f"{_PREFIX_USER}{user_id}"


async def check_user_rate_limit(
    redis: Redis,
    user_id: str,
    max_requests: int,
    window_seconds: int = _DEFAULT_WINDOW_SECONDS,
) -> bool:
    """Return True if the user is within their rate limit for the current window.

    The request is always recorded; the caller must check the return value and
    reject the message if False is returned.

    Args:
        redis: Active Redis client (decode_responses=True).
        user_id: Opaque string identifier for the user (e.g. str(telegram_user_id)).
        max_requests: Maximum allowed requests within ``window_seconds``.
        window_seconds: Rolling window duration in seconds (default 60).

    Returns:
        True if the request count (including this one) does not exceed
        ``max_requests``; False if the limit has been exceeded.
    """
    now = time.time()
    window_start = now - window_seconds
    key = _user_key(user_id)
    member = str(uuid.uuid4())

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, "-inf", window_start)
    pipe.zadd(key, {member: now})
    pipe.zcard(key)
    pipe.expire(key, window_seconds + 1)
    results: list[Any] = await pipe.execute()

    count: int = int(results[2])
    return count <= max_requests


async def check_global_rate_limit(
    redis: Redis,
    max_rps: int,
    window_seconds: int = 1,
) -> bool:
    """Return True if the global request rate is within the allowed RPS cap.

    Args:
        redis: Active Redis client.
        max_rps: Maximum requests allowed within ``window_seconds``.
        window_seconds: Window duration in seconds (default 1 for RPS).

    Returns:
        True if within limit; False if exceeded.
    """
    now = time.time()
    window_start = now - window_seconds
    key = _KEY_GLOBAL
    member = str(uuid.uuid4())

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, "-inf", window_start)
    pipe.zadd(key, {member: now})
    pipe.zcard(key)
    pipe.expire(key, window_seconds + 1)
    results: list[Any] = await pipe.execute()

    count: int = int(results[2])
    return count <= max_rps


async def get_user_request_count(
    redis: Redis,
    user_id: str,
    window_seconds: int = _DEFAULT_WINDOW_SECONDS,
) -> int:
    """Return the number of requests recorded for a user in the current window.

    Does not add a new entry.  Useful for monitoring and testing.
    """
    now = time.time()
    window_start = now - window_seconds
    key = _user_key(user_id)

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, "-inf", window_start)
    pipe.zcard(key)
    results: list[Any] = await pipe.execute()
    return int(results[1])
