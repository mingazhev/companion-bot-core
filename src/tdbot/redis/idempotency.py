"""Idempotency keys for Telegram update deduplication.

Telegram guarantees at-least-once delivery: the same ``update_id`` may arrive
multiple times when using webhooks (retries on 5xx) or after a polling
reconnect.  We record each ``update_id`` in Redis using an atomic SET NX EX
so that duplicate deliveries are detected and silently dropped by the ingress
layer.

Key layout
----------
``idempotency:update:{update_id}`` → literal ``"1"``, with a 24-hour TTL.

Telegram retry windows are in the order of minutes; 24 hours provides a large
safety margin while keeping memory usage bounded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis

_PREFIX = "idempotency:update:"
DEFAULT_TTL_SECONDS = 86_400  # 24 hours


def _key(update_id: int) -> str:
    return f"{_PREFIX}{update_id}"


async def mark_update_seen(
    redis: Redis[str],
    update_id: int,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> bool:
    """Atomically record *update_id* as processed.

    Uses ``SET NX EX`` for an atomic check-and-set: the key is only written if
    it does not already exist, so concurrent workers processing the same update
    will each get a deterministic True/False result.

    Args:
        redis: Active Redis client (decode_responses=True).
        update_id: Telegram ``Update.update_id`` value.
        ttl_seconds: Key expiry in seconds (default 86 400 = 24 hours).

    Returns:
        True  – first time this update_id has been seen; caller should process.
        False – already recorded; caller should skip (duplicate delivery).
    """
    result: bool | None = await redis.set(_key(update_id), "1", nx=True, ex=ttl_seconds)
    return result is not None


async def is_update_seen(
    redis: Redis[str],
    update_id: int,
) -> bool:
    """Return True if *update_id* has already been processed.

    Non-destructive read; does not modify TTL or create an entry.
    Useful for pre-flight checks in test helpers and monitoring.
    """
    count: int = await redis.exists(_key(update_id))
    return count > 0


async def clear_update_key(
    redis: Redis[str],
    update_id: int,
) -> None:
    """Delete the idempotency key for *update_id*.

    Intended for testing and administrative tooling only.  In production the
    key should expire naturally via its TTL.
    """
    await redis.delete(_key(update_id))
