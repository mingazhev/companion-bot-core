"""Unit tests for command handlers in tdbot.bot.handlers.

Handlers are called directly with mocked aiogram Message objects to verify
the response text and logging without a live Telegram connection.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tdbot.bot.handlers import (
    _VALID_TONES,
    cmd_delete_my_data,
    cmd_memory_compact_now,
    cmd_privacy,
    cmd_profile,
    cmd_reset_persona,
    cmd_set_persona,
    cmd_set_tone,
    cmd_start,
    handle_message,
)
from tdbot.db.models import User, UserProfile
from tdbot.prompt.snapshot_store import InMemorySnapshotStore

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_user(telegram_user_id: int = 42) -> User:
    user = User(telegram_user_id=telegram_user_id)
    user.id = uuid.uuid4()
    user.status = "active"
    user.locale = None
    user.timezone = None
    return user


def _make_message() -> AsyncMock:
    msg = AsyncMock()
    msg.answer = AsyncMock()
    return msg


def _make_command(args: str | None) -> MagicMock:
    cmd = MagicMock()
    cmd.args = args
    return cmd


def _make_profile_session(existing_profile: UserProfile | None = None) -> AsyncMock:
    """Session mock that supports _get_or_create_profile."""
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = existing_profile

    db_session = AsyncMock()
    db_session.add = MagicMock()
    db_session.execute = AsyncMock(return_value=select_result)
    return db_session


# --------------------------------------------------------------------------- #
# /start
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_start_replies() -> None:
    msg = _make_message()
    user = _make_user()
    await cmd_start(msg, user)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "/profile" in text
    assert "/set_tone" in text


# --------------------------------------------------------------------------- #
# /profile
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_profile_shows_telegram_id() -> None:
    msg = _make_message()
    user = _make_user(telegram_user_id=12345)
    await cmd_profile(msg, user)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "12345" in text


# --------------------------------------------------------------------------- #
# /set_tone
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_set_tone_valid() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="friendly")
    db_session = _make_profile_session()
    snapshot_store = InMemorySnapshotStore()
    await cmd_set_tone(msg, cmd, user, db_session, snapshot_store)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "friendly" in text


@pytest.mark.asyncio
async def test_set_tone_invalid_shows_valid_list() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="grumpy")
    await cmd_set_tone(msg, cmd, user, AsyncMock(), AsyncMock())
    text: str = msg.answer.call_args[0][0]
    assert "grumpy" in text
    # At least one valid tone should be mentioned
    assert any(t in text for t in _VALID_TONES)


@pytest.mark.asyncio
async def test_set_tone_no_args_shows_help() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args=None)
    await cmd_set_tone(msg, cmd, user, AsyncMock(), AsyncMock())
    text: str = msg.answer.call_args[0][0]
    assert "tone" in text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("tone", list(_VALID_TONES))
async def test_all_valid_tones_accepted(tone: str) -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args=tone)
    db_session = _make_profile_session()
    snapshot_store = InMemorySnapshotStore()
    await cmd_set_tone(msg, cmd, user, db_session, snapshot_store)
    text: str = msg.answer.call_args[0][0]
    assert tone in text


@pytest.mark.asyncio
async def test_set_tone_persists_profile_and_creates_snapshot() -> None:
    """Verify that /set_tone updates the profile and creates a prompt snapshot."""
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="playful")
    db_session = _make_profile_session()
    snapshot_store = InMemorySnapshotStore()

    await cmd_set_tone(msg, cmd, user, db_session, snapshot_store)

    # Profile was added to session (new user, no existing profile)
    db_session.add.assert_called_once()
    added_profile: UserProfile = db_session.add.call_args[0][0]
    assert added_profile.tone == "playful"

    # A new snapshot was saved and set active
    active = await snapshot_store.get_active(user.id)
    assert active is not None
    assert active.source == "user_command"
    assert "Tone: playful" in active.system_prompt


@pytest.mark.asyncio
async def test_set_tone_updates_existing_profile() -> None:
    """Verify that /set_tone updates an existing profile instead of creating new."""
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="casual")

    existing_profile = UserProfile(user_id=user.id, persona_name="Ada")
    existing_profile.tone = "friendly"
    db_session = _make_profile_session(existing_profile=existing_profile)
    snapshot_store = InMemorySnapshotStore()

    await cmd_set_tone(msg, cmd, user, db_session, snapshot_store)

    # Existing profile was updated (not a new one added)
    db_session.add.assert_not_called()
    assert existing_profile.tone == "casual"

    # Snapshot includes both persona name and new tone
    active = await snapshot_store.get_active(user.id)
    assert active is not None
    assert "Tone: casual" in active.system_prompt
    assert "Name: Ada" in active.system_prompt


# --------------------------------------------------------------------------- #
# /set_persona
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_set_persona_valid() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="Alex")
    db_session = _make_profile_session()
    snapshot_store = InMemorySnapshotStore()
    await cmd_set_persona(msg, cmd, user, db_session, snapshot_store)
    text: str = msg.answer.call_args[0][0]
    assert "Alex" in text


@pytest.mark.asyncio
async def test_set_persona_html_in_name_is_escaped() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="<b>Bold</b>")
    db_session = _make_profile_session()
    snapshot_store = InMemorySnapshotStore()
    await cmd_set_persona(msg, cmd, user, db_session, snapshot_store)
    text: str = msg.answer.call_args[0][0]
    assert "<b>" not in text
    assert "&lt;b&gt;" in text


@pytest.mark.asyncio
async def test_set_persona_empty_shows_help() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="")
    await cmd_set_persona(msg, cmd, user, AsyncMock(), AsyncMock())
    text: str = msg.answer.call_args[0][0]
    assert "persona" in text.lower()


@pytest.mark.asyncio
async def test_set_persona_too_long_rejected() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="A" * 65)
    await cmd_set_persona(msg, cmd, user, AsyncMock(), AsyncMock())
    text: str = msg.answer.call_args[0][0]
    assert "64" in text


@pytest.mark.asyncio
async def test_set_persona_persists_profile_and_creates_snapshot() -> None:
    """Verify that /set_persona updates the profile and creates a prompt snapshot."""
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="Nova")
    db_session = _make_profile_session()
    snapshot_store = InMemorySnapshotStore()

    await cmd_set_persona(msg, cmd, user, db_session, snapshot_store)

    db_session.add.assert_called_once()
    added_profile: UserProfile = db_session.add.call_args[0][0]
    assert added_profile.persona_name == "Nova"

    active = await snapshot_store.get_active(user.id)
    assert active is not None
    assert active.source == "user_command"
    assert "Name: Nova" in active.system_prompt


# --------------------------------------------------------------------------- #
# /memory_compact_now
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_memory_compact_now_replies() -> None:
    msg = _make_message()
    user = _make_user()
    redis = AsyncMock()
    redis.rpush = AsyncMock(return_value=1)
    await cmd_memory_compact_now(msg, user, redis)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "compaction" in text.lower()


@pytest.mark.asyncio
async def test_memory_compact_now_enqueues_refinement_job() -> None:
    msg = _make_message()
    user = _make_user()
    redis = AsyncMock()
    redis.rpush = AsyncMock(return_value=1)
    await cmd_memory_compact_now(msg, user, redis)
    redis.rpush.assert_called_once()
    import json

    call_args = redis.rpush.call_args
    queue_name = call_args[0][0]
    payload = json.loads(call_args[0][1])
    assert queue_name == "refinement_jobs"
    assert payload["trigger"] == "manual_compact"
    assert payload["user_id"] == str(user.id)


# --------------------------------------------------------------------------- #
# /reset_persona
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reset_persona_replies() -> None:
    msg = _make_message()
    user = _make_user()
    existing_profile = UserProfile(user_id=user.id, persona_name="Ada", tone="playful")
    db_session = _make_profile_session(existing_profile=existing_profile)
    snapshot_store = InMemorySnapshotStore()
    await cmd_reset_persona(msg, user, db_session, snapshot_store)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "been reset" in text.lower()


@pytest.mark.asyncio
async def test_reset_persona_clears_profile_and_creates_snapshot() -> None:
    """Verify that /reset_persona clears persona fields and creates a clean snapshot."""
    msg = _make_message()
    user = _make_user()

    existing_profile = UserProfile(user_id=user.id, persona_name="Ada", tone="playful")
    db_session = _make_profile_session(existing_profile=existing_profile)
    snapshot_store = InMemorySnapshotStore()

    await cmd_reset_persona(msg, user, db_session, snapshot_store)

    # Profile fields cleared
    assert existing_profile.persona_name is None
    assert existing_profile.tone is None

    # Snapshot has no persona section
    active = await snapshot_store.get_active(user.id)
    assert active is not None
    assert active.source == "user_command"
    assert "[Persona]" not in active.system_prompt


# --------------------------------------------------------------------------- #
# /privacy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_privacy_mentions_delete() -> None:
    msg = _make_message()
    await cmd_privacy(msg)
    text: str = msg.answer.call_args[0][0]
    assert "/delete_my_data" in text


# --------------------------------------------------------------------------- #
# /delete_my_data
# --------------------------------------------------------------------------- #


def _make_delete_session(user: User) -> AsyncMock:
    """Session mock where the SELECT existence check finds the user."""
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = user.id
    delete_result = MagicMock()

    db_session = AsyncMock()
    db_session.add = MagicMock()
    db_session.execute = AsyncMock(side_effect=[select_result, delete_result])
    return db_session


@pytest.mark.asyncio
async def test_delete_my_data_replies() -> None:
    msg = _make_message()
    user = _make_user()
    db_session = _make_delete_session(user)
    redis = AsyncMock()
    redis.delete = AsyncMock()
    snapshot_store = AsyncMock()
    await cmd_delete_my_data(msg, user, db_session, redis, snapshot_store)
    # Verify snapshot store cleanup
    snapshot_store.delete_for_user.assert_awaited_once_with(user.id)
    # Verify the DB statements were executed (SELECT + DELETE)
    assert db_session.execute.await_count == 2
    # Verify Redis keys were cleaned up
    redis.delete.assert_awaited_once()
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "deleted" in text.lower()


@pytest.mark.asyncio
async def test_delete_my_data_cleans_correct_redis_keys() -> None:
    msg = _make_message()
    user = _make_user(telegram_user_id=42)
    db_session = _make_delete_session(user)
    redis = AsyncMock()
    redis.delete = AsyncMock()
    snapshot_store = AsyncMock()
    await cmd_delete_my_data(msg, user, db_session, redis, snapshot_store)
    deleted_keys: tuple[str, ...] = redis.delete.call_args[0]
    user_id_str = str(user.id)
    # All internal-UUID-scoped keys must reference the user's UUID
    uuid_keys = [k for k in deleted_keys if user_id_str in k]
    assert len(uuid_keys) >= 1
    assert any("pending_change" in k for k in deleted_keys)
    assert any("activity_count" in k for k in deleted_keys)
    assert any("abuse:block" in k for k in deleted_keys)
    # Rate-limit key must reference the Telegram user ID (42), not the UUID
    assert any("rate_limit:user:42" in k for k in deleted_keys)


# --------------------------------------------------------------------------- #
# handle_message
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handle_message_sends_reply() -> None:
    msg = _make_message()
    msg.text = "Hello"
    user = _make_user()
    db_session = AsyncMock()
    redis = AsyncMock()
    snapshot_store = AsyncMock()
    chat_client = AsyncMock()
    settings = MagicMock()
    settings.chat_model = "gpt-4o-mini"
    settings.conversation_ttl_seconds = 604800
    settings.refinement_activity_threshold = 10

    with (
        patch("tdbot.bot.handlers.process_message", return_value="Hello back") as mock_process,
        patch("tdbot.bot.handlers.check_and_clear_user_notice", return_value=False),
    ):
        await handle_message(msg, user, db_session, redis, snapshot_store, chat_client, settings)

    mock_process.assert_awaited_once()
    msg.answer.assert_called_once_with("Hello back", parse_mode=None)


@pytest.mark.asyncio
async def test_handle_message_sends_profile_updated_notice_when_set() -> None:
    msg = _make_message()
    msg.text = "Hi"
    user = _make_user()
    db_session = AsyncMock()
    redis = AsyncMock()
    snapshot_store = AsyncMock()
    chat_client = AsyncMock()
    settings = MagicMock()
    settings.chat_model = "gpt-4o-mini"
    settings.conversation_ttl_seconds = 604800
    settings.refinement_activity_threshold = 10

    with patch("tdbot.bot.handlers.process_message", return_value="Reply text"), patch(
        "tdbot.bot.handlers.check_and_clear_user_notice", return_value=True
    ):
        await handle_message(msg, user, db_session, redis, snapshot_store, chat_client, settings)

    assert msg.answer.call_count == 2
    assert msg.answer.call_args_list[0][0][0] == "Reply text"
    assert "profile" in msg.answer.call_args_list[1][0][0].lower()


@pytest.mark.asyncio
async def test_handle_message_no_notice_when_not_set() -> None:
    msg = _make_message()
    msg.text = "Bye"
    user = _make_user()
    db_session = AsyncMock()
    redis = AsyncMock()
    snapshot_store = AsyncMock()
    chat_client = AsyncMock()
    settings = MagicMock()
    settings.chat_model = "gpt-4o-mini"
    settings.conversation_ttl_seconds = 604800
    settings.refinement_activity_threshold = 10

    with patch("tdbot.bot.handlers.process_message", return_value="Goodbye"), patch(
        "tdbot.bot.handlers.check_and_clear_user_notice", return_value=False
    ):
        await handle_message(msg, user, db_session, redis, snapshot_store, chat_client, settings)

    msg.answer.assert_called_once_with("Goodbye", parse_mode=None)
