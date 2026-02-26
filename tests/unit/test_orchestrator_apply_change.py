"""Unit tests for behavior change application in the orchestrator.

Verifies that detected behavior changes (tone, persona, skill add/remove) are
applied to the prompt snapshot when auto_applied or confirmed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import fakeredis.aioredis as fakeredis
import pytest

from tdbot.behavior.schemas import DetectionResult
from tdbot.inference.schemas import InferenceReply, SafetyFlags, TokenUsage
from tdbot.orchestrator.dialogue_state import PendingChange, set_pending_change
from tdbot.orchestrator.orchestrator import (
    _CHANGE_APPLIED_MSG,
    process_message,
)
from tdbot.prompt.schemas import SnapshotRecord
from tdbot.prompt.snapshot_store import InMemorySnapshotStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inference_reply(text: str = "OK") -> InferenceReply:
    return InferenceReply(
        reply=text,
        usage=TokenUsage(prompt_tokens=30, completion_tokens=12, total_tokens=42),
        safety_flags=SafetyFlags(
            content_filtered=False, refusal=False, finish_reason="stop"
        ),
    )


def _make_detection(
    intent: str = "normal_chat",
    risk_level: str = "low",
    confidence: float = 0.9,
    action: str = "pass_through",
) -> DetectionResult:
    return DetectionResult(
        intent=intent,  # type: ignore[arg-type]
        risk_level=risk_level,  # type: ignore[arg-type]
        confidence=confidence,
        action=action,  # type: ignore[arg-type]
    )


def _make_session() -> AsyncMock:
    """Minimal async session mock with profile query support."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    # For SELECT queries used by _get_or_create_profile
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=scalar_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


def _seed_snapshot(store: InMemorySnapshotStore, user_id: object) -> SnapshotRecord:
    """Create an initial snapshot in the store for testing."""
    import asyncio

    async def _seed() -> SnapshotRecord:
        version = await store.next_version(user_id)  # type: ignore[arg-type]
        record = SnapshotRecord(
            user_id=user_id,  # type: ignore[arg-type]
            version=version,
            system_prompt="You are a helpful, friendly companion.",
            skill_prompts_json={},
            source="initial",
        )
        await store.save(record)
        await store.set_active(user_id, record.id)  # type: ignore[arg-type]
        return record

    return asyncio.get_event_loop().run_until_complete(_seed())


# ---------------------------------------------------------------------------
# Auto-apply: tone_change updates snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_apply_tone_change_updates_snapshot() -> None:
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    session = _make_session()
    store = InMemorySnapshotStore()

    # Seed an initial snapshot
    version = await store.next_version(user_id)
    initial = SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt="You are a helpful, friendly companion.",
        skill_prompts_json={},
        source="initial",
    )
    await store.save(initial)
    await store.set_active(user_id, initial.id)

    with patch(
        "tdbot.orchestrator.orchestrator.classify",
        return_value=_make_detection(
            intent="tone_change", risk_level="low", action="auto_apply"
        ),
    ), patch(
        "tdbot.orchestrator.orchestrator.generate_reply",
        return_value=_make_inference_reply("Sure, I'll be more playful!"),
    ):
        await process_message(
            user_id=user_id,
            message_text="Be more playful please",
            session=session,
            snapshot_store=store,
            redis=redis,
            chat_client=MagicMock(),
        )

    # Snapshot should have been updated with the new tone
    active = await store.get_active(user_id)
    assert active is not None
    assert active.version > initial.version
    assert active.source == "behavior_change"
    assert "Tone: playful" in active.system_prompt


@pytest.mark.asyncio
async def test_auto_apply_tone_change_no_extraction_still_records_event() -> None:
    """When tone cannot be extracted, event is still recorded but snapshot unchanged."""
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    session = _make_session()
    store = InMemorySnapshotStore()

    version = await store.next_version(user_id)
    initial = SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt="You are a helpful, friendly companion.",
        skill_prompts_json={},
        source="initial",
    )
    await store.save(initial)
    await store.set_active(user_id, initial.id)

    with patch(
        "tdbot.orchestrator.orchestrator.classify",
        return_value=_make_detection(
            intent="tone_change", risk_level="low", action="auto_apply"
        ),
    ), patch(
        "tdbot.orchestrator.orchestrator.generate_reply",
        return_value=_make_inference_reply("OK"),
    ):
        await process_message(
            user_id=user_id,
            message_text="Change your style to something different",
            session=session,
            snapshot_store=store,
            redis=redis,
            chat_client=MagicMock(),
        )

    # Snapshot should still be the initial one (extraction failed)
    active = await store.get_active(user_id)
    assert active is not None
    assert active.version == initial.version


# ---------------------------------------------------------------------------
# Confirmed: persona_change updates snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmed_persona_change_updates_snapshot() -> None:
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    session = _make_session()
    store = InMemorySnapshotStore()

    version = await store.next_version(user_id)
    initial = SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt="You are a helpful, friendly companion.",
        skill_prompts_json={},
        source="initial",
    )
    await store.save(initial)
    await store.set_active(user_id, initial.id)

    # Seed a pending change for persona
    detection = _make_detection(
        intent="persona_change", risk_level="medium", action="confirm"
    )
    pending = PendingChange(
        detection_result=detection,
        original_message="You are now Alex",
    )
    await set_pending_change(redis, str(user_id), pending)

    reply = await process_message(
        user_id=user_id,
        message_text="yes",
        session=session,
        snapshot_store=store,
        redis=redis,
        chat_client=MagicMock(),
    )

    assert reply == _CHANGE_APPLIED_MSG

    # Snapshot should have been updated with the persona name
    active = await store.get_active(user_id)
    assert active is not None
    assert active.version > initial.version
    assert active.source == "behavior_change"
    assert "Name: Alex" in active.system_prompt


# ---------------------------------------------------------------------------
# Auto-apply: skill_add_prompt updates snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_apply_skill_add_updates_snapshot() -> None:
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    session = _make_session()
    store = InMemorySnapshotStore()

    version = await store.next_version(user_id)
    initial = SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt="You are a helpful, friendly companion.",
        skill_prompts_json={},
        source="initial",
    )
    await store.save(initial)
    await store.set_active(user_id, initial.id)

    with patch(
        "tdbot.orchestrator.orchestrator.classify",
        return_value=_make_detection(
            intent="skill_add_prompt", risk_level="low", action="auto_apply"
        ),
    ), patch(
        "tdbot.orchestrator.orchestrator.generate_reply",
        return_value=_make_inference_reply("I can help with cooking!"),
    ):
        await process_message(
            user_id=user_id,
            message_text="Add skill for cooking",
            session=session,
            snapshot_store=store,
            redis=redis,
            chat_client=MagicMock(),
        )

    active = await store.get_active(user_id)
    assert active is not None
    assert active.version > initial.version
    assert "cooking" in active.skill_prompts_json
    assert "[Skill: cooking]" in active.system_prompt


# ---------------------------------------------------------------------------
# Auto-apply: skill_remove updates snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_apply_skill_remove_updates_snapshot() -> None:
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    session = _make_session()
    store = InMemorySnapshotStore()

    # Seed a snapshot with a cooking skill
    version = await store.next_version(user_id)
    initial = SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt=(
            "You are a helpful, friendly companion.\n\n---\n\n"
            "[Skill: cooking]\nAssist the user with cooking-related questions and tasks."
        ),
        skill_prompts_json={
            "cooking": "Assist the user with cooking-related questions and tasks."
        },
        source="initial",
    )
    await store.save(initial)
    await store.set_active(user_id, initial.id)

    with patch(
        "tdbot.orchestrator.orchestrator.classify",
        return_value=_make_detection(
            intent="skill_remove", risk_level="low", action="auto_apply"
        ),
    ), patch(
        "tdbot.orchestrator.orchestrator.generate_reply",
        return_value=_make_inference_reply("OK, I'll stop helping with cooking."),
    ):
        await process_message(
            user_id=user_id,
            message_text="Stop helping me with cooking",
            session=session,
            snapshot_store=store,
            redis=redis,
            chat_client=MagicMock(),
        )

    active = await store.get_active(user_id)
    assert active is not None
    assert active.version > initial.version
    assert "cooking" not in active.skill_prompts_json
    assert "[Skill: cooking]" not in active.system_prompt
