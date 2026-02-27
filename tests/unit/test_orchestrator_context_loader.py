"""Unit tests for tdbot.orchestrator.context_loader."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from tdbot.inference.schemas import ChatMessage, UserContext
from tdbot.orchestrator.context_loader import load_recent_messages, load_user_context
from tdbot.prompt.schemas import SnapshotRecord
from tdbot.prompt.snapshot_store import InMemorySnapshotStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conv_message(role: str, content: str) -> Any:
    """Return a minimal mock object with role and content, mimicking a ConversationMessage row.

    Using MagicMock avoids SQLAlchemy instrumentation issues when building
    test objects without a real database session.
    """
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg


def _make_session(rows: list[Any]) -> AsyncMock:
    """Return a mocked AsyncSession that yields *rows* on execute()."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    return session


def _make_snapshot(user_id: Any, system_prompt: str = "Custom prompt.") -> SnapshotRecord:
    return SnapshotRecord(
        user_id=user_id,
        version=1,
        system_prompt=system_prompt,
        source="initial",
    )


# ---------------------------------------------------------------------------
# load_recent_messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_recent_messages_returns_chat_messages() -> None:
    rows = [
        _make_conv_message("assistant", "Hi!"),
        _make_conv_message("user", "Hello"),
    ]
    # DB returns newest-first; loader reverses to oldest-first
    session = _make_session(rows)
    user_id = uuid4()

    result = await load_recent_messages(session, user_id)

    assert len(result) == 2
    assert all(isinstance(m, ChatMessage) for m in result)
    # After reversal, oldest (index 1 in rows) comes first
    assert result[0].role == "user"
    assert result[0].content == "Hello"
    assert result[1].role == "assistant"
    assert result[1].content == "Hi!"


@pytest.mark.asyncio
async def test_load_recent_messages_empty_returns_empty_list() -> None:
    session = _make_session([])
    result = await load_recent_messages(session, uuid4())
    assert result == []


@pytest.mark.asyncio
async def test_load_recent_messages_calls_session_execute() -> None:
    session = _make_session([])
    user_id = uuid4()

    await load_recent_messages(session, user_id)

    session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_load_recent_messages_preserves_role_and_content() -> None:
    rows = [_make_conv_message("user", "What is the weather?")]
    session = _make_session(rows)

    result = await load_recent_messages(session, uuid4())

    assert result[0].role == "user"
    assert result[0].content == "What is the weather?"


# ---------------------------------------------------------------------------
# load_user_context — with active snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_user_context_uses_active_snapshot_prompt() -> None:
    user_id = uuid4()
    store = InMemorySnapshotStore()
    snap = _make_snapshot(user_id, system_prompt="You are Alex.")
    await store.save(snap)
    await store.set_active(user_id, snap.id)

    session = _make_session([])
    ctx = await load_user_context(session, store, user_id)

    assert isinstance(ctx, UserContext)
    assert ctx.system_prompt == "You are Alex."
    assert ctx.user_id == str(user_id)


@pytest.mark.asyncio
async def test_load_user_context_falls_back_to_default_when_no_snapshot() -> None:
    user_id = uuid4()
    store = InMemorySnapshotStore()  # empty — no snapshot
    session = _make_session([])

    ctx = await load_user_context(session, store, user_id)

    lower = ctx.system_prompt.lower()
    assert "helpful" in lower or "companion" in lower or "дружелюб" in lower or "компаньон" in lower


@pytest.mark.asyncio
async def test_load_user_context_includes_recent_history() -> None:
    user_id = uuid4()
    store = InMemorySnapshotStore()
    snap = _make_snapshot(user_id)
    await store.save(snap)
    await store.set_active(user_id, snap.id)

    rows = [
        _make_conv_message("assistant", "Hi!"),
        _make_conv_message("user", "Hello"),
    ]
    session = _make_session(rows)

    ctx = await load_user_context(session, store, user_id)

    assert len(ctx.conversation_history) == 2


@pytest.mark.asyncio
async def test_load_user_context_respects_max_tokens_param() -> None:
    user_id = uuid4()
    store = InMemorySnapshotStore()
    session = _make_session([])

    ctx = await load_user_context(session, store, user_id, max_tokens=512)

    assert ctx.max_tokens == 512


@pytest.mark.asyncio
async def test_load_user_context_empty_history_when_no_messages() -> None:
    user_id = uuid4()
    store = InMemorySnapshotStore()
    session = _make_session([])

    ctx = await load_user_context(session, store, user_id)

    assert ctx.conversation_history == []
