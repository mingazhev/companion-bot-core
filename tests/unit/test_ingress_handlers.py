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
from tdbot.db.models import User

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
    await cmd_set_tone(msg, cmd, user)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "friendly" in text


@pytest.mark.asyncio
async def test_set_tone_invalid_shows_valid_list() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="grumpy")
    await cmd_set_tone(msg, cmd, user)
    text: str = msg.answer.call_args[0][0]
    assert "grumpy" in text
    # At least one valid tone should be mentioned
    assert any(t in text for t in _VALID_TONES)


@pytest.mark.asyncio
async def test_set_tone_no_args_shows_help() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args=None)
    await cmd_set_tone(msg, cmd, user)
    text: str = msg.answer.call_args[0][0]
    assert "tone" in text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("tone", list(_VALID_TONES))
async def test_all_valid_tones_accepted(tone: str) -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args=tone)
    await cmd_set_tone(msg, cmd, user)
    text: str = msg.answer.call_args[0][0]
    assert tone in text


# --------------------------------------------------------------------------- #
# /set_persona
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_set_persona_valid() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="Alex")
    await cmd_set_persona(msg, cmd, user)
    text: str = msg.answer.call_args[0][0]
    assert "Alex" in text


@pytest.mark.asyncio
async def test_set_persona_html_in_name_is_escaped() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="<b>Bold</b>")
    await cmd_set_persona(msg, cmd, user)
    text: str = msg.answer.call_args[0][0]
    assert "<b>" not in text
    assert "&lt;b&gt;" in text


@pytest.mark.asyncio
async def test_set_persona_empty_shows_help() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="")
    await cmd_set_persona(msg, cmd, user)
    text: str = msg.answer.call_args[0][0]
    assert "persona" in text.lower()


@pytest.mark.asyncio
async def test_set_persona_too_long_rejected() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="A" * 65)
    await cmd_set_persona(msg, cmd, user)
    text: str = msg.answer.call_args[0][0]
    assert "64" in text


# --------------------------------------------------------------------------- #
# /memory_compact_now
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_memory_compact_now_replies() -> None:
    msg = _make_message()
    user = _make_user()
    await cmd_memory_compact_now(msg, user)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "compaction" in text.lower()


# --------------------------------------------------------------------------- #
# /reset_persona
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reset_persona_replies() -> None:
    msg = _make_message()
    user = _make_user()
    await cmd_reset_persona(msg, user)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "been reset" in text.lower()


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


@pytest.mark.asyncio
async def test_delete_my_data_replies() -> None:
    msg = _make_message()
    user = _make_user()
    db_session = AsyncMock()
    db_session.add = MagicMock()
    db_session.execute = AsyncMock()
    redis = AsyncMock()
    redis.delete = AsyncMock()
    await cmd_delete_my_data(msg, user, db_session, redis)
    # Verify the DB delete statement was actually executed
    db_session.execute.assert_awaited()
    # Verify Redis keys were cleaned up
    redis.delete.assert_awaited_once()
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "deleted" in text.lower()


@pytest.mark.asyncio
async def test_delete_my_data_cleans_correct_redis_keys() -> None:
    msg = _make_message()
    user = _make_user(telegram_user_id=42)
    db_session = AsyncMock()
    db_session.add = MagicMock()
    db_session.execute = AsyncMock()
    redis = AsyncMock()
    redis.delete = AsyncMock()
    await cmd_delete_my_data(msg, user, db_session, redis)
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
