"""Unit tests for UX transformation features.

Tests for: /memory, /remember, /forget, /help, reworked /start,
/refresh_memory, /settings, /personas, /skills, onboarding callbacks,
continuity hints, and suggestion hints.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from companion_bot_core.bot.handlers import (
    DEEP_PERSONAS,
    cb_onboard_interest,
    cb_onboard_tone,
    cb_skill_toggle,
    cb_tone_pick,
    cmd_forget,
    cmd_help,
    cmd_memory,
    cmd_personas,
    cmd_refresh_memory,
    cmd_remember,
    cmd_settings,
    cmd_skills,
    cmd_start,
)
from companion_bot_core.db.models import User, UserProfile
from companion_bot_core.prompt.helpers import (
    add_fact_to_profile,
    extract_memory_sections,
    remove_fact_from_profile,
)
from companion_bot_core.prompt.merge_builder import build_system_prompt
from companion_bot_core.prompt.schemas import PromptComponents, SnapshotRecord
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


def _make_callback(data: str) -> AsyncMock:
    cb = AsyncMock()
    cb.data = data
    cb.answer = AsyncMock()
    cb.message = AsyncMock()
    cb.message.edit_text = AsyncMock()
    return cb


def _make_profile_session(
    existing_profile: UserProfile | None = None,
    *,
    user_id: uuid.UUID | None = None,
) -> AsyncMock:
    profile = existing_profile or UserProfile(user_id=user_id or uuid.uuid4())
    select_result = MagicMock()
    select_result.scalar_one.return_value = profile
    select_result.scalar_one_or_none.return_value = profile

    db_session = AsyncMock()
    db_session.info = {}
    db_session.add = MagicMock()
    db_session.execute = AsyncMock(return_value=select_result)
    return db_session


def _make_empty_profile_session() -> AsyncMock:
    """Session that returns None for profile queries (new user)."""
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = None
    select_result.scalar_one.return_value = UserProfile(user_id=uuid.uuid4())

    db_session = AsyncMock()
    db_session.info = {}
    db_session.add = MagicMock()
    db_session.execute = AsyncMock(return_value=select_result)
    return db_session


def _make_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.delete = AsyncMock()
    redis.rpush = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.getset = AsyncMock(return_value=None)
    redis.expire = AsyncMock()
    return redis


async def _make_snapshot_with_profile(
    user_id: uuid.UUID,
) -> tuple[InMemorySnapshotStore, SnapshotRecord]:
    store = InMemorySnapshotStore()
    components = PromptComponents(
        base_system_template="You are a helpful AI.",
        persona_segment="Имя пользователя: TestBot\nTone: friendly",
        skill_packs={"code_assistant": "Help with code."},
        long_term_profile="Likes coffee\nStudies math",
    )
    system_prompt = build_system_prompt(components)
    record = SnapshotRecord(
        user_id=user_id,
        version=1,
        system_prompt=system_prompt,
        skill_prompts_json={"code_assistant": "Help with code."},
        source="initial",
    )
    await store.save(record)
    await store.set_active(user_id, record.id)
    return store, record


# --------------------------------------------------------------------------- #
# extract_memory_sections
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_extract_memory_sections_with_data() -> None:
    user_id = uuid.uuid4()
    store, _ = await _make_snapshot_with_profile(user_id)
    snapshot = await store.get_active(user_id)
    sections = extract_memory_sections(snapshot)
    assert sections["persona"] == "TestBot"
    assert sections["tone"] == "friendly"
    assert "code_assistant" in sections["skills"]
    assert "Likes coffee" in sections["long_term_profile"]


def test_extract_memory_sections_none() -> None:
    sections = extract_memory_sections(None)
    assert sections["persona"] == ""
    assert sections["tone"] == ""
    assert sections["skills"] == ""
    assert sections["long_term_profile"] == ""


# --------------------------------------------------------------------------- #
# add_fact_to_profile / remove_fact_from_profile
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_add_fact_to_profile() -> None:
    user_id = uuid.uuid4()
    store, _ = await _make_snapshot_with_profile(user_id)
    await add_fact_to_profile(store, user_id, "Enjoys hiking")
    snapshot = await store.get_active(user_id)
    assert snapshot is not None
    assert "Enjoys hiking" in snapshot.system_prompt
    assert "Likes coffee" in snapshot.system_prompt


@pytest.mark.asyncio
async def test_add_fact_to_empty_profile() -> None:
    user_id = uuid.uuid4()
    store = InMemorySnapshotStore()
    await add_fact_to_profile(store, user_id, "First fact")
    snapshot = await store.get_active(user_id)
    assert snapshot is not None
    assert "First fact" in snapshot.system_prompt


@pytest.mark.asyncio
async def test_remove_fact_from_profile() -> None:
    user_id = uuid.uuid4()
    store, _ = await _make_snapshot_with_profile(user_id)
    removed = await remove_fact_from_profile(store, user_id, "coffee")
    assert removed is not None
    assert "coffee" in removed.lower()
    snapshot = await store.get_active(user_id)
    assert snapshot is not None
    assert "coffee" not in snapshot.system_prompt.lower()
    assert "math" in snapshot.system_prompt.lower()


@pytest.mark.asyncio
async def test_remove_fact_not_found() -> None:
    user_id = uuid.uuid4()
    store, _ = await _make_snapshot_with_profile(user_id)
    removed = await remove_fact_from_profile(store, user_id, "nonexistent")
    assert removed is None


@pytest.mark.asyncio
async def test_remove_fact_no_snapshot() -> None:
    user_id = uuid.uuid4()
    store = InMemorySnapshotStore()
    removed = await remove_fact_from_profile(store, user_id, "anything")
    assert removed is None


# --------------------------------------------------------------------------- #
# /memory command
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cmd_memory_with_data() -> None:
    msg = _make_message()
    user = _make_user()
    store, _ = await _make_snapshot_with_profile(user.id)
    await cmd_memory(msg, user, store)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "TestBot" in text or "помню" in text.lower() or "remember" in text.lower()


@pytest.mark.asyncio
async def test_cmd_memory_empty() -> None:
    msg = _make_message()
    user = _make_user()
    store = InMemorySnapshotStore()
    await cmd_memory(msg, user, store)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "не знаю" in text.lower() or "don't know" in text.lower()


# --------------------------------------------------------------------------- #
# /remember command
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cmd_remember_saves_fact() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="I love Python")
    store = InMemorySnapshotStore()
    db_session = _make_profile_session()

    await cmd_remember(msg, cmd, user, db_session, store)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "Python" in text

    snapshot = await store.get_active(user.id)
    assert snapshot is not None
    assert "I love Python" in snapshot.system_prompt


@pytest.mark.asyncio
async def test_cmd_remember_empty_shows_help() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="")
    store = InMemorySnapshotStore()
    await cmd_remember(msg, cmd, user, AsyncMock(), store)
    text: str = msg.answer.call_args[0][0]
    assert "/remember" in text


# --------------------------------------------------------------------------- #
# /forget command
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cmd_forget_removes_matching_fact() -> None:
    msg = _make_message()
    user = _make_user()
    store, _ = await _make_snapshot_with_profile(user.id)
    cmd = _make_command(args="coffee")
    db_session = _make_profile_session()

    await cmd_forget(msg, cmd, user, db_session, store)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    # Should report what was forgotten
    assert "coffee" in text.lower() or "забыл" in text.lower()


@pytest.mark.asyncio
async def test_cmd_forget_not_found() -> None:
    msg = _make_message()
    user = _make_user()
    store, _ = await _make_snapshot_with_profile(user.id)
    cmd = _make_command(args="nonexistent_thing")
    db_session = _make_profile_session()

    await cmd_forget(msg, cmd, user, db_session, store)
    text: str = msg.answer.call_args[0][0]
    assert "не нашёл" in text.lower() or "couldn't find" in text.lower()


@pytest.mark.asyncio
async def test_cmd_forget_empty_shows_help() -> None:
    msg = _make_message()
    user = _make_user()
    cmd = _make_command(args="")
    store = InMemorySnapshotStore()
    await cmd_forget(msg, cmd, user, AsyncMock(), store)
    text: str = msg.answer.call_args[0][0]
    assert "/forget" in text


# --------------------------------------------------------------------------- #
# /help command
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cmd_help_lists_commands() -> None:
    msg = _make_message()
    user = _make_user()
    await cmd_help(msg, user)
    text: str = msg.answer.call_args[0][0]
    assert "/memory" in text
    assert "/remember" in text
    assert "/forget" in text
    assert "/profile" in text


# --------------------------------------------------------------------------- #
# /start (reworked)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cmd_start_new_user() -> None:
    msg = _make_message()
    user = _make_user()
    db_session = _make_empty_profile_session()
    store = InMemorySnapshotStore()
    redis = AsyncMock()

    await cmd_start(msg, user, db_session, store, redis)
    # cmd_start sends 2 messages for new users (welcome + name prompt)
    assert msg.answer.call_count >= 1
    text: str = msg.answer.call_args_list[0][0][0]
    # New user gets value-prop message
    assert "компаньон" in text.lower() or "companion" in text.lower()
    # Redis should store onboarding state
    redis.set.assert_called_once()


@pytest.mark.asyncio
async def test_cmd_start_returning_user_with_name() -> None:
    msg = _make_message()
    user = _make_user()
    profile = UserProfile(user_id=user.id, persona_name="Alice", tone="friendly")
    db_session = _make_profile_session(existing_profile=profile)
    store = InMemorySnapshotStore()
    redis = AsyncMock()

    await cmd_start(msg, user, db_session, store, redis)
    text: str = msg.answer.call_args[0][0]
    assert "Alice" in text


# --------------------------------------------------------------------------- #
# /refresh_memory (renamed /memory_compact_now)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cmd_refresh_memory_enqueues_job() -> None:
    msg = _make_message()
    user = _make_user()
    redis = _make_redis()
    await cmd_refresh_memory(msg, user, redis)
    msg.answer.assert_called_once()
    text: str = msg.answer.call_args[0][0]
    assert "просмотрю" in text.lower() or "review" in text.lower()
    redis.rpush.assert_called_once()


@pytest.mark.asyncio
async def test_cmd_refresh_memory_in_progress() -> None:
    msg = _make_message()
    user = _make_user()
    redis = _make_redis()
    redis.set = AsyncMock(return_value=None)  # Guard not acquired
    await cmd_refresh_memory(msg, user, redis)
    text: str = msg.answer.call_args[0][0]
    assert "уже" in text.lower() or "already" in text.lower()


# --------------------------------------------------------------------------- #
# /settings command
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cmd_settings_shows_keyboard() -> None:
    msg = _make_message()
    user = _make_user()
    await cmd_settings(msg, user)
    msg.answer.assert_called_once()
    call_kwargs = msg.answer.call_args[1]
    assert "reply_markup" in call_kwargs
    kb = call_kwargs["reply_markup"]
    # Should have inline keyboard rows
    assert len(kb.inline_keyboard) >= 2


# --------------------------------------------------------------------------- #
# Tone picker callback
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cb_tone_pick_sets_tone() -> None:
    cb = _make_callback("tone:friendly")
    user = _make_user()
    db_session = _make_profile_session(user_id=user.id)
    store = InMemorySnapshotStore()

    await cb_tone_pick(cb, user, db_session, store)
    cb.answer.assert_called_once()
    snapshot = await store.get_active(user.id)
    assert snapshot is not None
    assert "friendly" in snapshot.system_prompt.lower()


# --------------------------------------------------------------------------- #
# /personas command
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cmd_personas_shows_keyboard() -> None:
    msg = _make_message()
    user = _make_user()
    await cmd_personas(msg, user)
    msg.answer.assert_called_once()
    call_kwargs = msg.answer.call_args[1]
    assert "reply_markup" in call_kwargs
    kb = call_kwargs["reply_markup"]
    # Should have buttons for deep + seed personas
    total_buttons = sum(len(row) for row in kb.inline_keyboard)
    assert total_buttons >= len(DEEP_PERSONAS) + 3  # at least deep + some seed


# --------------------------------------------------------------------------- #
# /skills command
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cmd_skills_shows_catalog() -> None:
    msg = _make_message()
    user = _make_user()
    store = InMemorySnapshotStore()
    await cmd_skills(msg, user, store)
    msg.answer.assert_called_once()
    call_kwargs = msg.answer.call_args[1]
    assert "reply_markup" in call_kwargs


@pytest.mark.asyncio
async def test_cb_skill_toggle_adds_skill() -> None:
    cb = _make_callback("skill_toggle:code_assistant")
    user = _make_user()
    store = InMemorySnapshotStore()
    db_session = _make_profile_session()

    await cb_skill_toggle(cb, user, db_session, store)
    cb.answer.assert_called_once()
    snapshot = await store.get_active(user.id)
    assert snapshot is not None
    assert "code_assistant" in snapshot.skill_prompts_json


@pytest.mark.asyncio
async def test_cb_skill_toggle_removes_skill() -> None:
    user = _make_user()
    store, _ = await _make_snapshot_with_profile(user.id)
    cb = _make_callback("skill_toggle:code_assistant")
    db_session = _make_profile_session()

    await cb_skill_toggle(cb, user, db_session, store)
    snapshot = await store.get_active(user.id)
    assert snapshot is not None
    assert "code_assistant" not in snapshot.skill_prompts_json


# --------------------------------------------------------------------------- #
# Onboarding callbacks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_onboarding_interest_stores_state() -> None:
    cb = _make_callback("onboard_interest:tech")
    user = _make_user()
    redis = _make_redis()

    await cb_onboard_interest(cb, user, redis)
    cb.answer.assert_called_once()
    # Verify Redis state was stored
    redis.set.assert_called()
    call_args = redis.set.call_args[0]
    state = json.loads(call_args[1])
    assert state["interest"] == "tech"


@pytest.mark.asyncio
async def test_onboarding_tone_completes_setup() -> None:
    cb = _make_callback("onboard_tone:friendly")
    user = _make_user()
    redis = _make_redis()
    # Simulate existing onboarding state
    redis.get = AsyncMock(return_value=json.dumps({"name": "Alice", "interest": "tech"}))

    db_session = _make_profile_session(user_id=user.id)
    store = InMemorySnapshotStore()

    await cb_onboard_tone(cb, user, db_session, store, redis)
    cb.answer.assert_called_once()
    # Redis state should be cleaned up
    redis.delete.assert_called()
    # Snapshot should be created
    snapshot = await store.get_active(user.id)
    assert snapshot is not None
