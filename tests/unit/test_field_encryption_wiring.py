"""Unit tests for field encryption wiring into DB read/write paths.

Verifies that:
- Conversation message content is encrypted before persistence and decrypted
  on read when a FieldEncryptor with encryption enabled is provided.
- User profile fields (persona_name, tone) are encrypted before writes and
  decrypted for prompt building.
- Legacy unencrypted rows are handled gracefully via decrypt_safe.
- When no encryptor is provided (None), values pass through unchanged.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import fakeredis.aioredis as fakeredis
import pytest
from cryptography.fernet import Fernet

from companion_bot_core.behavior.schemas import DetectionResult
from companion_bot_core.inference.schemas import InferenceReply, SafetyFlags, TokenUsage
from companion_bot_core.orchestrator.context_loader import load_recent_messages
from companion_bot_core.orchestrator.dialogue_state import PendingChange, set_pending_change
from companion_bot_core.orchestrator.orchestrator import (
    _CHANGE_APPLIED_MSG,
    process_message,
)
from companion_bot_core.privacy.field_encryption import FieldEncryptor
from companion_bot_core.prompt.schemas import SnapshotRecord
from companion_bot_core.prompt.snapshot_store import InMemorySnapshotStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEY = Fernet.generate_key()


def _make_encryptor(enabled: bool = True) -> FieldEncryptor:
    if enabled:
        return FieldEncryptor(_KEY, enabled=True)
    return FieldEncryptor(None, enabled=False)


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


def _make_session(
    profile: Any = None,
) -> AsyncMock:
    """Minimal async session mock with profile query support.

    When *profile* is provided, ``scalar_one()`` returns it so that the
    upsert-then-SELECT pattern in ``get_or_create_profile`` works correctly.
    """
    from companion_bot_core.db.models import UserProfile

    scalar_result = MagicMock()
    scalar_result.scalars.return_value.all.return_value = []
    # Default profile for tests that go through the profile creation path.
    scalar_result.scalar_one.return_value = profile or UserProfile(user_id=uuid4())
    session = AsyncMock()
    session.execute = AsyncMock(return_value=scalar_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    # begin_nested must be a sync callable returning an AsyncMock so that
    # `async with session.begin_nested()` works as an async context manager.
    session.begin_nested = MagicMock(return_value=AsyncMock())
    return session


def _make_conv_row(role: str, content: str) -> Any:
    """Return a mock object mimicking a ConversationMessage row."""
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg


def _make_session_with_rows(rows: list[Any]) -> AsyncMock:
    """Return a mocked AsyncSession that yields *rows* on execute()."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    return session


# ---------------------------------------------------------------------------
# Conversation message encryption (write path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_messages_encrypts_content_when_enabled() -> None:
    """Conversation message content is encrypted before being added to the session."""
    enc = _make_encryptor(enabled=True)
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    session = _make_session()
    store = InMemorySnapshotStore()

    with patch(
        "companion_bot_core.orchestrator.orchestrator.classify",
        return_value=_make_detection(action="pass_through"),
    ), patch(
        "companion_bot_core.orchestrator.orchestrator.generate_reply",
        return_value=_make_inference_reply("Hello there!"),
    ):
        await process_message(
            user_id=user_id,
            message_text="Hi bot",
            session=session,
            snapshot_store=store,
            redis=redis,
            chat_client=MagicMock(),
            encryptor=enc,
        )

    # Two add() calls: user message and assistant message
    assert session.add.call_count == 2
    user_msg = session.add.call_args_list[0][0][0]
    asst_msg = session.add.call_args_list[1][0][0]

    # Content should NOT be plaintext — it should be encrypted
    assert user_msg.content != "Hi bot"
    assert asst_msg.content != "Hello there!"

    # Decrypting should recover the original plaintext
    assert enc.decrypt(user_msg.content) == "Hi bot"
    assert enc.decrypt(asst_msg.content) == "Hello there!"


@pytest.mark.asyncio
async def test_persist_messages_passthrough_when_no_encryptor() -> None:
    """Without an encryptor, content is stored as plaintext (backward compatible)."""
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    session = _make_session()
    store = InMemorySnapshotStore()

    with patch(
        "companion_bot_core.orchestrator.orchestrator.classify",
        return_value=_make_detection(action="pass_through"),
    ), patch(
        "companion_bot_core.orchestrator.orchestrator.generate_reply",
        return_value=_make_inference_reply("Hello there!"),
    ):
        await process_message(
            user_id=user_id,
            message_text="Hi bot",
            session=session,
            snapshot_store=store,
            redis=redis,
            chat_client=MagicMock(),
            # No encryptor passed — default None
        )

    user_msg = session.add.call_args_list[0][0][0]
    asst_msg = session.add.call_args_list[1][0][0]
    assert user_msg.content == "Hi bot"
    assert asst_msg.content == "Hello there!"


# ---------------------------------------------------------------------------
# Conversation message decryption (read path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_recent_messages_decrypts_content() -> None:
    """Encrypted content in DB rows is decrypted when loading recent messages."""
    enc = _make_encryptor(enabled=True)
    encrypted_content = enc.encrypt("Hello world")
    rows = [_make_conv_row("user", encrypted_content)]
    session = _make_session_with_rows(rows)

    result = await load_recent_messages(session, uuid4(), encryptor=enc)

    assert len(result) == 1
    assert result[0].content == "Hello world"


@pytest.mark.asyncio
async def test_load_recent_messages_handles_legacy_unencrypted_rows() -> None:
    """Legacy unencrypted rows are returned as-is via decrypt_safe fallback."""
    enc = _make_encryptor(enabled=True)
    # Plain text that is NOT a valid Fernet token
    rows = [_make_conv_row("user", "plain old text")]
    session = _make_session_with_rows(rows)

    result = await load_recent_messages(session, uuid4(), encryptor=enc)

    assert len(result) == 1
    # decrypt_safe returns the original content as the default
    assert result[0].content == "plain old text"


@pytest.mark.asyncio
async def test_load_recent_messages_passthrough_without_encryptor() -> None:
    """Without an encryptor, content is returned unchanged (backward compatible)."""
    rows = [_make_conv_row("assistant", "Hi!")]
    session = _make_session_with_rows(rows)

    result = await load_recent_messages(session, uuid4())

    assert result[0].content == "Hi!"


# ---------------------------------------------------------------------------
# Profile field encryption (tone_change via auto_apply)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_apply_tone_change_encrypts_profile_tone() -> None:
    """When a tone change is auto-applied, the profile tone is stored encrypted."""
    enc = _make_encryptor(enabled=True)
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    from companion_bot_core.db.models import UserProfile

    profile = UserProfile(user_id=user_id)
    session = _make_session(profile=profile)
    store = InMemorySnapshotStore()

    # Seed initial snapshot
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
        "companion_bot_core.orchestrator.orchestrator.classify",
        return_value=_make_detection(
            intent="tone_change", risk_level="low", action="auto_apply"
        ),
    ), patch(
        "companion_bot_core.orchestrator.orchestrator.generate_reply",
        return_value=_make_inference_reply("Sure!"),
    ):
        await process_message(
            user_id=user_id,
            message_text="Be more playful please",
            session=session,
            snapshot_store=store,
            redis=redis,
            chat_client=MagicMock(),
            encryptor=enc,
        )

    # The tone field should be encrypted, not plaintext
    assert profile.tone is not None
    assert profile.tone != "playful"
    assert enc.decrypt(profile.tone) == "playful"

    # But the prompt snapshot should contain the raw (decrypted) tone
    active = await store.get_active(user_id)
    assert active is not None
    assert "Tone: playful" in active.system_prompt


# ---------------------------------------------------------------------------
# Profile field encryption (persona_change via confirmation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmed_persona_change_encrypts_profile_name() -> None:
    """When a persona change is confirmed, the profile persona_name is stored encrypted."""
    enc = _make_encryptor(enabled=True)
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    from companion_bot_core.db.models import UserProfile

    profile = UserProfile(user_id=user_id)
    session = _make_session(profile=profile)
    store = InMemorySnapshotStore()

    # Seed initial snapshot
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

    # Seed a pending persona change
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
        encryptor=enc,
    )

    assert reply == _CHANGE_APPLIED_MSG

    # persona_name should be encrypted
    assert profile.persona_name is not None
    assert profile.persona_name != "Alex"
    assert enc.decrypt(profile.persona_name) == "Alex"

    # But the prompt snapshot should contain the raw name
    active = await store.get_active(user_id)
    assert active is not None
    assert "Имя пользователя: Alex" in active.system_prompt


# ---------------------------------------------------------------------------
# Disabled encryptor passes through (backward compat)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_encryptor_passes_through_profile_fields() -> None:
    """A disabled encryptor stores profile fields as plaintext."""
    enc = _make_encryptor(enabled=False)
    user_id = uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    from companion_bot_core.db.models import UserProfile

    profile = UserProfile(user_id=user_id)
    session = _make_session(profile=profile)
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
        "companion_bot_core.orchestrator.orchestrator.classify",
        return_value=_make_detection(
            intent="tone_change", risk_level="low", action="auto_apply"
        ),
    ), patch(
        "companion_bot_core.orchestrator.orchestrator.generate_reply",
        return_value=_make_inference_reply("Sure!"),
    ):
        await process_message(
            user_id=user_id,
            message_text="Be more playful please",
            session=session,
            snapshot_store=store,
            redis=redis,
            chat_client=MagicMock(),
            encryptor=enc,
        )

    # With disabled encryptor, tone is stored as plaintext
    assert profile.tone == "playful"


# ---------------------------------------------------------------------------
# Round-trip: encrypt on write, decrypt on read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encrypted_message_roundtrip_through_context_loader() -> None:
    """Messages encrypted during persist are correctly decrypted by the context loader."""
    enc = _make_encryptor(enabled=True)
    user_id = uuid4()

    # Simulate an encrypted DB row
    encrypted_user = enc.encrypt("What is 2+2?")
    encrypted_asst = enc.encrypt("2+2 equals 4.")

    rows = [
        _make_conv_row("assistant", encrypted_asst),
        _make_conv_row("user", encrypted_user),
    ]
    session = _make_session_with_rows(rows)

    result = await load_recent_messages(session, user_id, encryptor=enc)

    # Oldest first after reversal
    assert result[0].content == "What is 2+2?"
    assert result[0].role == "user"
    assert result[1].content == "2+2 equals 4."
    assert result[1].role == "assistant"
