"""Short-lived prompt context cache.

Caches the active prompt snapshot for a user in Redis with a configurable TTL.
This avoids a database round-trip on every chat message: the orchestrator reads
the cached snapshot first and falls back to PostgreSQL only on a miss.

Key layout
----------
``prompt_cache:{user_id}`` → JSON-serialised snapshot dict.

The entry is invalidated explicitly whenever the active prompt snapshot changes
(e.g. after a refinement job completes or the user applies a config change).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis.asyncio import Redis

_PREFIX = "prompt_cache:"
DEFAULT_TTL_SECONDS = 300  # 5 minutes


def _key(user_id: str) -> str:
    return f"{_PREFIX}{user_id}"


async def cache_prompt(
    redis: Redis[str],
    user_id: str,
    prompt_data: dict[str, Any],
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """Persist the prompt context for *user_id* with an expiry.

    Args:
        redis: Active Redis client (decode_responses=True).
        user_id: Opaque string identifier for the user.
        prompt_data: Serialisable dict representing the prompt snapshot.
        ttl_seconds: Time-to-live in seconds (default 300).
    """
    await redis.set(_key(user_id), json.dumps(prompt_data), ex=ttl_seconds)


async def get_cached_prompt(
    redis: Redis[str],
    user_id: str,
) -> dict[str, Any] | None:
    """Return the cached prompt context for *user_id*, or None on a miss.

    Returns:
        Deserialised prompt dict if the cache entry exists; None otherwise.
    """
    raw: str | None = await redis.get(_key(user_id))
    if raw is None:
        return None
    return json.loads(raw)  # type: ignore[no-any-return]


async def invalidate_prompt_cache(
    redis: Redis[str],
    user_id: str,
) -> None:
    """Remove the cached prompt for *user_id*.

    Should be called whenever the active prompt snapshot changes so the next
    chat request loads the fresh version from the database.
    """
    await redis.delete(_key(user_id))


async def extend_prompt_cache_ttl(
    redis: Redis[str],
    user_id: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> bool:
    """Reset the TTL on an existing cache entry without touching its value.

    Args:
        redis: Active Redis client.
        user_id: Opaque string identifier for the user.
        ttl_seconds: New TTL in seconds.

    Returns:
        True if the key existed and its TTL was updated; False if it had already
        expired or never existed.
    """
    result: bool = await redis.expire(_key(user_id), ttl_seconds)
    return result
