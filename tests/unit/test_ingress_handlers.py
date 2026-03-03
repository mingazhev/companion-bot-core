"""Unit tests for command handlers in companion_bot_core.bot.handlers.

Handlers are called directly with mocked aiogram Message objects to verify
the response text and logging without a live Telegram connection.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from companion_bot_core.behavior.extractor import VALID_TONES
from companion_bot_core.bot.handlers import (
    cb_delete_data_no,
    cb_delete_data_yes,
    cb_reset_no,
    cb_reset_yes,
    cmd_delete_my_data,
    cmd_memory_compact_now,
    cmd_privacy,
    cmd_profile,
    cmd_refresh_memory,
    cmd_reset,
    cmd_reset_persona,
    cmd_set_language,
    cmd_set_persona,
    cmd_set_tone,
    cmd_start,
    handle_message,
)
from companion_bot_core.db.models import User, UserProfile
from companion_bot_core.prompt.snapshot_store import InMemorySnapshotStore

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


def _make_profile_session(
    existing_profile: UserProfile | None = None,
    *,
    user_id: uuid.UUID | None = None,
) -> AsyncMock:
    """Session mock that supports get_or_create_profile (upsert-then-SELECT).

    Always returns a real ``UserProfile`` from ``scalar_one()`` so the
    handler code can set attributes on it.  ``info`` is a real dict so that
    ``session.info.setdefault(...)`` used by the deferred lock/write helpers
    works correctly.
    """
    profile = existing_profile or UserProfile(user_id=user_id or uuid.uuid4())
    select_result = MagicMock()
    select_result.scalar_one.return_value = profile

    db_session = AsyncMock()
    db_session.info = {}  # real dict — required by defer_lock_release / deferred writes
    db_session.add = MagicMock()
    db_session.execute = AsyncMock(return_value=select_result)
    return db_session


# --------------------------------------------------------------------------- #
# /start
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_start_replies_new_user() -> None:
    """New user (no profile) gets a warm welcome message."""
    msg = _make_message()
    user = _make_user()
    db_session = _make_profile_session()
    # Override: scalar_one_or_none returns None (no existing profile)
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = None
    db_session.execute = AsyncMock(return_value=select_result)
    snapshot_store = InMemorySnapshotStore()
    redis = AsyncMock()
    await cmd_start(msg, user, db_session, snapshot_store, redis)
    # New user gets welcome message + onboarding name prompt
    assert msg.answer.call_count >= 1
    first_text: str = msg.answer.call_args_list[0][0][0]
    assert "companion" in first_text.lower() or "компаньон" in first_text.lower()
    # Redis should have onboarding state set
    redis.set.assert_called_once()


@pytest.mark.asyncio
async def test_start_replies_returning_user() -> None:
    """Returning user (has profile with persona) gets a personalised greeting."""
    msg = _make_message()
    user = _make_user()
    profile = UserProfile(user_id=user.id, persona_name="Ada")
    profile.tone = "friendly"
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = profile
    db_session = AsyncMock()
    db_session.info = {}
    db_session.execute = AsyncMock(return_value=select_result)
    snapshot_store = InMemorySnapshotStore()
    redis = AsyncMock()
    await cmd_start(msg, user, db_session, snapshot_store, redis)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "Ada" in text or "возвращ" in text.lower() or "welcome back" in text.lower()


# --------------------------------------------------------------------------- #
# /profile
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_profile_shows_telegram_id() -> None:
    msg = _make_message()
    user = _make_user(telegram_user_id=12345)
    db_session = _make_profile_session()
    await cmd_profile(msg, user, db_session)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "12345" in text


# --------------------------------------------------------------------------- #
# /set_tone
# --------------------------------------------------------------------------- #


def _make_redis(*, lock_acquired: bool = True) -> AsyncMock:
    """Redis mock with controllable SET NX (lock) behaviour."""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True if lock_acquired else None)
    redis.delete = AsyncMock()
    redis.eval = AsyncMock(return_value=1)
    redis.rpush = AsyncMock(return_value=1)
    return redis


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
    assert any(t in text for t in VALID_TONES)


@pytest.mark.asyncio
async def test_set_tone_no_args_shows_help() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args=None)
    await cmd_set_tone(msg, cmd, user, AsyncMock(), AsyncMock())
    text: str = msg.answer.call_args[0][0]
    assert "tone" in text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("tone", list(VALID_TONES))
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
    profile = UserProfile(user_id=user.id)
    db_session = _make_profile_session(existing_profile=profile)
    snapshot_store = InMemorySnapshotStore()

    await cmd_set_tone(msg, cmd, user, db_session, snapshot_store)

    # Profile tone was updated via upsert-then-SELECT
    assert profile.tone == "playful"

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

    # Existing profile was updated
    assert existing_profile.tone == "casual"

    # Snapshot includes both persona name and new tone
    active = await snapshot_store.get_active(user.id)
    assert active is not None
    assert "Tone: casual" in active.system_prompt
    assert "Name: Ada" in active.system_prompt


@pytest.mark.asyncio
async def test_set_tone_rejects_when_lock_held() -> None:
    """When the profile write lock is already held, /set_tone returns an error."""
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="friendly")
    redis = _make_redis(lock_acquired=False)

    await cmd_set_tone(msg, cmd, user, AsyncMock(), InMemorySnapshotStore(), redis=redis)

    text: str = msg.answer.call_args[0][0]
    lower = text.lower()
    assert "in progress" in lower or "обновление профиля" in lower
    # Lock was not acquired so neither DEL nor eval-unlock must be called.
    redis.delete.assert_not_called()
    redis.eval.assert_not_called()


@pytest.mark.asyncio
async def test_set_tone_acquires_and_defers_lock_release() -> None:
    """Verify the profile write lock is acquired with a unique token then
    deferred for post-commit release (not released inside the handler).

    The lock release is stored in session.info so IngressMiddleware can
    flush it after the DB transaction commits and the Redis active pointer
    is updated — eliminating the race where a concurrent request acquires
    the lock before A's changes are fully visible.
    """
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="playful")
    db_session = _make_profile_session()
    snapshot_store = InMemorySnapshotStore()
    redis = _make_redis(lock_acquired=True)

    await cmd_set_tone(msg, cmd, user, db_session, snapshot_store, redis=redis)

    expected_key = f"profile:write:{user.id}"
    # Lock must be acquired with the correct key, nx=True, ex=30, and a
    # non-static token (not the literal "1").
    redis.set.assert_awaited_once()
    set_args, set_kwargs = redis.set.call_args
    assert set_args[0] == expected_key
    assert isinstance(set_args[1], str) and len(set_args[1]) > 0
    assert set_args[1] != "1", "token must be unique, not the static value '1'"
    assert set_kwargs.get("nx") is True
    assert set_kwargs.get("ex") == 120
    # Lock release must be deferred — NOT released inside the handler.
    redis.eval.assert_not_awaited()
    redis.delete.assert_not_awaited()
    # Lock info must be stored in session.info for post-commit release by
    # IngressMiddleware via flush_deferred_lock_releases.
    deferred = db_session.info.get("_profile_lock_releases", [])
    assert len(deferred) == 1
    assert deferred[0] == (expected_key, set_args[1])


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
    profile = UserProfile(user_id=user.id)
    db_session = _make_profile_session(existing_profile=profile)
    snapshot_store = InMemorySnapshotStore()

    await cmd_set_persona(msg, cmd, user, db_session, snapshot_store)

    # Profile persona_name was updated via upsert-then-SELECT
    assert profile.persona_name == "Nova"

    active = await snapshot_store.get_active(user.id)
    assert active is not None
    assert active.source == "user_command"
    assert "Name: Nova" in active.system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "Hello\nWorld",
        "Name\tWith\tTabs",
        "Null\x00Byte",
        "CR\rInMiddle",
    ],
    ids=["newline", "tab", "null_byte", "carriage_return"],
)
async def test_set_persona_control_chars_rejected(name: str) -> None:
    """Control characters in persona name must be rejected (prompt injection defense)."""
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args=name)
    await cmd_set_persona(msg, cmd, user, AsyncMock(), AsyncMock())
    text: str = msg.answer.call_args[0][0]
    lower = text.lower()
    assert "control characters" in lower or "управляющие символы" in lower


@pytest.mark.asyncio
async def test_set_persona_rejects_when_lock_held() -> None:
    """When the profile write lock is held, /set_persona returns an error."""
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="Nova")
    redis = _make_redis(lock_acquired=False)

    await cmd_set_persona(msg, cmd, user, AsyncMock(), InMemorySnapshotStore(), redis=redis)

    text: str = msg.answer.call_args[0][0]
    lower = text.lower()
    assert "in progress" in lower or "обновление профиля" in lower
    redis.delete.assert_not_called()
    redis.eval.assert_not_called()


@pytest.mark.asyncio
async def test_reset_persona_rejects_when_lock_held() -> None:
    """When the profile write lock is held, /reset_persona returns an error."""
    msg = _make_message()
    user = _make_user()
    redis = _make_redis(lock_acquired=False)

    await cmd_reset_persona(msg, user, AsyncMock(), InMemorySnapshotStore(), redis=redis)

    text: str = msg.answer.call_args[0][0]
    lower = text.lower()
    assert "in progress" in lower or "обновление профиля" in lower
    redis.delete.assert_not_called()
    redis.eval.assert_not_called()


# --------------------------------------------------------------------------- #
# /memory_compact_now
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_refresh_memory_replies() -> None:
    msg = _make_message()
    user = _make_user()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)  # guard acquired
    redis.rpush = AsyncMock(return_value=1)
    await cmd_refresh_memory(msg, user, redis)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    lower = text.lower()
    # New natural wording: "review conversations" / "обновлю"
    assert "review" in lower or "просмотрю" in lower or "обновлю" in lower


@pytest.mark.asyncio
async def test_refresh_memory_enqueues_refinement_job() -> None:
    msg = _make_message()
    user = _make_user()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)  # guard acquired
    redis.rpush = AsyncMock(return_value=1)
    await cmd_refresh_memory(msg, user, redis)
    redis.rpush.assert_called_once()
    import json

    call_args = redis.rpush.call_args
    queue_name = call_args[0][0]
    payload = json.loads(call_args[0][1])
    assert queue_name == "refinement_jobs"
    assert payload["trigger"] == "manual_compact"
    assert payload["user_id"] == str(user.id)


@pytest.mark.asyncio
async def test_memory_compact_now_delegates_to_refresh_memory() -> None:
    """Legacy /memory_compact_now still works via delegation."""
    msg = _make_message()
    user = _make_user()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.rpush = AsyncMock(return_value=1)
    await cmd_memory_compact_now(msg, user, redis)
    msg.answer.assert_called_once()
    redis.rpush.assert_called_once()


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
    lower = text.lower()
    assert "been reset" in lower or "сброш" in lower


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
async def test_delete_my_data_shows_confirmation() -> None:
    """The /delete_my_data command should show a confirmation prompt, not delete immediately."""
    msg = _make_message()
    user = _make_user()
    redis = AsyncMock()
    redis.set = AsyncMock()
    await cmd_delete_my_data(msg, user, redis)
    # Should set a Redis confirmation guard.
    redis.set.assert_awaited_once()
    guard_key = redis.set.call_args[0][0]
    assert "delete_confirm" in guard_key
    # Should show a message with inline keyboard.
    msg.answer.assert_called_once()
    call_kwargs = msg.answer.call_args
    assert call_kwargs.kwargs.get("reply_markup") is not None


@pytest.mark.asyncio
async def test_delete_my_data_confirm_yes_deletes() -> None:
    """Pressing 'Yes' on the confirmation should delete all data."""
    user = _make_user(telegram_user_id=42)
    db_session = _make_delete_session(user)
    redis = AsyncMock()
    redis.getdel = AsyncMock(return_value="1")
    redis.delete = AsyncMock()
    snapshot_store = AsyncMock()
    callback = AsyncMock()
    callback.message = AsyncMock()
    callback.message.edit_text = AsyncMock()
    await cb_delete_data_yes(callback, user, db_session, redis, snapshot_store)
    # Verify snapshot store cleanup.
    snapshot_store.delete_for_user.assert_awaited_once_with(user.id)
    # Verify the DB statements were executed (SELECT + DELETE).
    assert db_session.execute.await_count == 2
    # Verify Redis keys were cleaned up.
    redis.delete.assert_awaited_once()
    # Verify the message was updated.
    callback.message.edit_text.assert_awaited_once()
    text: str = callback.message.edit_text.call_args[0][0]
    lower = text.lower()
    assert "удален" in lower or "deleted" in lower


@pytest.mark.asyncio
async def test_delete_my_data_confirm_expired() -> None:
    """Pressing 'Yes' after TTL expiry should show an expiry message."""
    user = _make_user()
    db_session = _make_delete_session(user)
    redis = AsyncMock()
    redis.getdel = AsyncMock(return_value=None)
    snapshot_store = AsyncMock()
    callback = AsyncMock()
    await cb_delete_data_yes(callback, user, db_session, redis, snapshot_store)
    callback.answer.assert_awaited_once()
    alert_text: str = callback.answer.call_args[0][0]
    assert "истекло" in alert_text.lower() or "expired" in alert_text.lower()
    # No deletion should happen.
    snapshot_store.delete_for_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_my_data_confirm_no_cancels() -> None:
    """Pressing 'No' should cancel deletion and clear the guard."""
    user = _make_user()
    redis = AsyncMock()
    redis.delete = AsyncMock()
    callback = AsyncMock()
    callback.message = AsyncMock()
    callback.message.edit_text = AsyncMock()
    await cb_delete_data_no(callback, user, redis)
    redis.delete.assert_awaited_once()
    callback.message.edit_text.assert_awaited_once()
    text: str = callback.message.edit_text.call_args[0][0]
    lower = text.lower()
    assert "отмен" in lower or "cancel" in lower


@pytest.mark.asyncio
async def test_delete_my_data_cleans_correct_redis_keys() -> None:
    user = _make_user(telegram_user_id=42)
    db_session = _make_delete_session(user)
    redis = AsyncMock()
    redis.getdel = AsyncMock(return_value="1")
    redis.delete = AsyncMock()
    snapshot_store = AsyncMock()
    callback = AsyncMock()
    callback.message = AsyncMock()
    callback.message.edit_text = AsyncMock()
    await cb_delete_data_yes(callback, user, db_session, redis, snapshot_store)
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
    redis.get = AsyncMock(return_value=None)  # no pending change
    snapshot_store = AsyncMock()
    chat_client = AsyncMock()
    settings = MagicMock()
    settings.chat_model = "gpt-4o-mini"
    settings.conversation_ttl_seconds = 604800
    settings.refinement_activity_threshold = 10

    with (
        patch(
            "companion_bot_core.bot.handlers.process_message",
            return_value="Hello back",
        ) as mock_process,
        patch("companion_bot_core.bot.handlers.check_and_clear_user_notice", return_value=None),
    ):
        await handle_message(msg, user, db_session, redis, snapshot_store, chat_client, settings)

    mock_process.assert_awaited_once()
    msg.answer.assert_called_once_with("Hello back", parse_mode=None, reply_markup=None)


@pytest.mark.asyncio
async def test_handle_message_silently_consumes_notice_when_set() -> None:
    msg = _make_message()
    msg.text = "Hi"
    user = _make_user()
    db_session = AsyncMock()
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)  # no pending change
    snapshot_store = AsyncMock()
    chat_client = AsyncMock()
    settings = MagicMock()
    settings.chat_model = "gpt-4o-mini"
    settings.conversation_ttl_seconds = 604800
    settings.refinement_activity_threshold = 10

    with patch("companion_bot_core.bot.handlers.process_message", return_value="Reply text"), patch(
        "companion_bot_core.bot.handlers.check_and_clear_user_notice", return_value={}
    ):
        await handle_message(msg, user, db_session, redis, snapshot_store, chat_client, settings)

    # Notice is silently consumed — reply contains only the model response.
    msg.answer.assert_called_once()
    sent_text: str = msg.answer.call_args[0][0]
    assert sent_text == "Reply text"


@pytest.mark.asyncio
async def test_set_language_updates_locale() -> None:
    msg = _make_message()
    user = _make_user()
    db_session = AsyncMock()
    cmd = _make_command(args="en")

    await cmd_set_language(msg, cmd, user, db_session)

    assert user.locale == "en"
    db_session.flush.assert_awaited_once()
    text: str = msg.answer.call_args[0][0]
    assert "english" in text.lower()


@pytest.mark.asyncio
async def test_handle_message_no_notice_when_not_set() -> None:
    msg = _make_message()
    msg.text = "Bye"
    user = _make_user()
    db_session = AsyncMock()
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    snapshot_store = AsyncMock()
    chat_client = AsyncMock()
    settings = MagicMock()
    settings.chat_model = "gpt-4o-mini"
    settings.conversation_ttl_seconds = 604800
    settings.refinement_activity_threshold = 10

    with patch("companion_bot_core.bot.handlers.process_message", return_value="Goodbye"), patch(
        "companion_bot_core.bot.handlers.check_and_clear_user_notice", return_value=None
    ):
        await handle_message(msg, user, db_session, redis, snapshot_store, chat_client, settings)

    msg.answer.assert_called_once_with("Goodbye", parse_mode=None, reply_markup=None)


# --------------------------------------------------------------------------- #
# /reset
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reset_shows_confirmation() -> None:
    """The /reset command should show a confirmation prompt."""
    msg = _make_message()
    user = _make_user()
    redis = AsyncMock()
    redis.set = AsyncMock()
    await cmd_reset(msg, user, redis)
    redis.set.assert_awaited_once()
    guard_key = redis.set.call_args[0][0]
    assert "reset_confirm" in guard_key
    msg.answer.assert_called_once()
    call_kwargs = msg.answer.call_args
    assert call_kwargs.kwargs.get("reply_markup") is not None


@pytest.mark.asyncio
async def test_reset_yes_deletes_child_data_and_starts_onboarding() -> None:
    """Pressing 'Yes' deletes child tables, clears Redis, restarts onboarding."""
    user = _make_user(telegram_user_id=42)
    db_session = AsyncMock()
    db_session.execute = AsyncMock()
    db_session.flush = AsyncMock()
    redis = AsyncMock()
    redis.getdel = AsyncMock(return_value="1")
    redis.delete = AsyncMock()
    redis.set = AsyncMock()
    snapshot_store = AsyncMock()
    callback = AsyncMock()
    callback.message = AsyncMock()
    callback.message.edit_text = AsyncMock()
    callback.message.answer = AsyncMock()

    await cb_reset_yes(callback, user, db_session, redis, snapshot_store)

    # 6 child-table DELETEs
    assert db_session.execute.await_count == 6
    db_session.flush.assert_awaited_once()
    # Snapshot store Redis pointers cleaned.
    snapshot_store.delete_for_user.assert_awaited_once_with(user.id)
    # Redis bulk delete called.
    redis.delete.assert_awaited_once()
    # Onboarding state set in Redis.
    redis.set.assert_awaited_once()
    onboard_key = redis.set.call_args[0][0]
    assert "onboarding" in onboard_key
    # Confirmation message edited.
    callback.message.edit_text.assert_awaited_once()
    text: str = callback.message.edit_text.call_args[0][0]
    lower = text.lower()
    assert "удален" in lower or "erase" in lower or "fresh" in lower
    # Onboarding step 1 message sent.
    callback.message.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_yes_expired_guard() -> None:
    """Pressing 'Yes' after TTL expiry should show an expiry message."""
    user = _make_user()
    db_session = AsyncMock()
    redis = AsyncMock()
    redis.getdel = AsyncMock(return_value=None)
    snapshot_store = AsyncMock()
    callback = AsyncMock()
    await cb_reset_yes(callback, user, db_session, redis, snapshot_store)
    callback.answer.assert_awaited_once()
    alert_text: str = callback.answer.call_args[0][0]
    assert "истекло" in alert_text.lower() or "expired" in alert_text.lower()
    snapshot_store.delete_for_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_no_cancels() -> None:
    """Pressing 'No' should cancel the reset and clear the guard."""
    user = _make_user()
    redis = AsyncMock()
    redis.delete = AsyncMock()
    callback = AsyncMock()
    callback.message = AsyncMock()
    callback.message.edit_text = AsyncMock()
    await cb_reset_no(callback, user, redis)
    redis.delete.assert_awaited_once()
    callback.message.edit_text.assert_awaited_once()
    text: str = callback.message.edit_text.call_args[0][0]
    lower = text.lower()
    assert "отмен" in lower or "cancel" in lower
