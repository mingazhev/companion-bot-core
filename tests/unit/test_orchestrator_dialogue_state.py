"""Unit tests for tdbot.orchestrator.dialogue_state."""

from __future__ import annotations

import fakeredis.aioredis as fakeredis
import pytest

from tdbot.behavior.schemas import DetectionResult
from tdbot.orchestrator.dialogue_state import (
    PendingChange,
    clear_pending_change,
    get_pending_change,
    set_pending_change,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(
    intent: str = "persona_change",
    risk_level: str = "medium",
    confidence: float = 0.8,
    action: str = "confirm",
) -> DetectionResult:
    return DetectionResult(
        intent=intent,  # type: ignore[arg-type]
        risk_level=risk_level,  # type: ignore[arg-type]
        confidence=confidence,
        action=action,  # type: ignore[arg-type]
        clarification_question=None,
    )


def _make_pending(original: str = "Make me more formal") -> PendingChange:
    return PendingChange(
        detection_result=_make_detection(),
        original_message=original,
    )


# ---------------------------------------------------------------------------
# PendingChange serialisation
# ---------------------------------------------------------------------------


def test_pending_change_json_round_trips() -> None:
    original = "Change my name to Alex"
    detection = _make_detection(intent="persona_change", confidence=0.75)
    pending = PendingChange(detection_result=detection, original_message=original)

    json_bytes = pending.model_dump_json()
    restored = PendingChange.model_validate_json(json_bytes)

    assert restored.original_message == original
    assert restored.detection_result.intent == "persona_change"
    assert restored.detection_result.confidence == pytest.approx(0.75)
    assert restored.detection_result.action == "confirm"


def test_pending_change_model_dump_preserves_all_fields() -> None:
    detection = _make_detection(
        intent="tone_change",
        risk_level="low",
        confidence=0.9,
        action="auto_apply",
    )
    pending = PendingChange(detection_result=detection, original_message="be friendlier")
    restored = PendingChange.model_validate(pending.model_dump())

    assert restored.detection_result.intent == "tone_change"
    assert restored.detection_result.risk_level == "low"
    assert restored.detection_result.action == "auto_apply"


# ---------------------------------------------------------------------------
# Redis operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_and_get_pending_change() -> None:
    redis = fakeredis.FakeRedis(decode_responses=True)
    user_id = "user-123"
    pending = _make_pending()

    await set_pending_change(redis, user_id, pending)
    result = await get_pending_change(redis, user_id)

    assert result is not None
    assert result.original_message == pending.original_message
    assert result.detection_result.intent == pending.detection_result.intent


@pytest.mark.asyncio
async def test_get_pending_change_returns_none_when_absent() -> None:
    redis = fakeredis.FakeRedis(decode_responses=True)
    result = await get_pending_change(redis, "no-such-user")
    assert result is None


@pytest.mark.asyncio
async def test_clear_pending_change_removes_key() -> None:
    redis = fakeredis.FakeRedis(decode_responses=True)
    user_id = "user-456"
    await set_pending_change(redis, user_id, _make_pending())

    await clear_pending_change(redis, user_id)

    result = await get_pending_change(redis, user_id)
    assert result is None


@pytest.mark.asyncio
async def test_clear_pending_change_is_idempotent() -> None:
    """Clearing a non-existent key must not raise."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    await clear_pending_change(redis, "ghost-user")


@pytest.mark.asyncio
async def test_set_pending_change_applies_ttl() -> None:
    redis = fakeredis.FakeRedis(decode_responses=True)
    user_id = "user-789"
    custom_ttl = 60

    await set_pending_change(redis, user_id, _make_pending(), ttl=custom_ttl)

    ttl = await redis.ttl(f"pending_change:{user_id}")
    assert 0 < ttl <= custom_ttl


@pytest.mark.asyncio
async def test_pending_change_isolated_per_user() -> None:
    """Each user_id maps to its own pending change slot."""
    redis = fakeredis.FakeRedis(decode_responses=True)

    await set_pending_change(redis, "user-A", _make_pending("message A"))
    await set_pending_change(redis, "user-B", _make_pending("message B"))

    result_a = await get_pending_change(redis, "user-A")
    result_b = await get_pending_change(redis, "user-B")

    assert result_a is not None and result_a.original_message == "message A"
    assert result_b is not None and result_b.original_message == "message B"

    await clear_pending_change(redis, "user-A")

    assert await get_pending_change(redis, "user-A") is None
    assert await get_pending_change(redis, "user-B") is not None
