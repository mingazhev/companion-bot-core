"""Confirmation dialogue state for medium-risk configuration changes.

When the behavior change detector returns ``action="confirm"`` the orchestrator
stores a :class:`PendingChange` in Redis keyed by user ID and asks the user to
confirm or cancel.  On the user's next message the orchestrator checks this key
first before classifying intent.

Redis key schema
----------------
``pending_change:<user_id>``
    JSON-serialised ``PendingChange``; TTL = ``_TTL_SECONDS`` (5 minutes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from tdbot.behavior.schemas import DetectionResult

if TYPE_CHECKING:
    from redis.asyncio import Redis

_KEY_PREFIX = "pending_change"
_TTL_SECONDS = 300  # 5 minutes


class PendingChange(BaseModel):
    """A medium-risk detection result awaiting user confirmation."""

    detection_result: DetectionResult
    original_message: str


def _key(user_id: str) -> str:
    return f"{_KEY_PREFIX}:{user_id}"


async def set_pending_change(
    redis: Redis,
    user_id: str,
    change: PendingChange,
    ttl: int = _TTL_SECONDS,
) -> None:
    """Persist *change* for *user_id* with an expiry of *ttl* seconds."""
    await redis.set(_key(user_id), change.model_dump_json(), ex=ttl)


async def get_pending_change(redis: Redis, user_id: str) -> PendingChange | None:
    """Return the pending change for *user_id*, or ``None`` if none exists."""
    raw = await redis.get(_key(user_id))
    if raw is None:
        return None
    return PendingChange.model_validate_json(raw)


async def clear_pending_change(redis: Redis, user_id: str) -> None:
    """Remove the pending change key for *user_id* (idempotent)."""
    await redis.delete(_key(user_id))
