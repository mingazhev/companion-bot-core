"""Redis connection factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import redis.asyncio as aioredis

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from tdbot.config import Settings


async def create_redis_pool(settings: Settings) -> Redis:
    """Create an async Redis client backed by a connection pool.

    The client is configured with decode_responses=True so all keys and values
    are returned as plain strings rather than bytes.
    """
    return aioredis.from_url(
        settings.redis_url.get_secret_value(),
        encoding="utf-8",
        decode_responses=True,
    )


async def close_redis_pool(redis: Redis) -> None:
    """Close the Redis client and release all pooled connections."""
    await redis.aclose()
