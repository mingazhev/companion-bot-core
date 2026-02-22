"""Unit tests for tdbot.bot.users — get_or_create_user helper."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from tdbot.bot.users import get_or_create_user
from tdbot.db.models import User


def _make_session(existing_user: User | None) -> MagicMock:
    """Return a mock AsyncSession with scalar_one_or_none returning *existing_user*.

    Uses MagicMock for the session (matching SQLAlchemy's sync methods like .add())
    and an AsyncMock only for .execute() which is async.
    """
    session = MagicMock()
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = existing_user
    session.execute = AsyncMock(return_value=execute_result)
    session.flush = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_returns_existing_user() -> None:
    """If the user already exists, return it without inserting."""
    existing = User(telegram_user_id=111)
    existing.id = uuid.uuid4()
    session = _make_session(existing)

    result = await get_or_create_user(session, telegram_user_id=111)
    assert result is existing
    session.add.assert_not_called()
    session.flush.assert_not_called()


@pytest.mark.asyncio
async def test_creates_user_when_not_found() -> None:
    """If no user exists, insert a new one and flush."""
    session = _make_session(None)

    result = await get_or_create_user(session, telegram_user_id=999)
    assert result.telegram_user_id == 999
    session.add.assert_called_once_with(result)
    session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_created_user_has_correct_telegram_id() -> None:
    """Newly created user has the telegram_user_id we passed in."""
    session = _make_session(None)
    result = await get_or_create_user(session, telegram_user_id=42)
    assert result.telegram_user_id == 42
