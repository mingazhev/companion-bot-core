"""Unit tests for companion_bot_core.bot.users — get_or_create_user helper."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from companion_bot_core.bot.users import get_or_create_user
from companion_bot_core.db.models import User


def _make_session(returned_user: User) -> MagicMock:
    """Return a mock AsyncSession that yields *returned_user* on the final SELECT.

    The new implementation issues:
      1. INSERT … ON CONFLICT DO NOTHING  (first execute call)
      2. SELECT … WHERE telegram_user_id = ?  (second execute call)

    The first execute returns a generic result (its value is unused).
    The second execute returns a result whose scalar_one() is *returned_user*.
    """
    session = MagicMock()

    insert_result = MagicMock()  # result of the upsert — not inspected

    select_result = MagicMock()
    select_result.scalar_one.return_value = returned_user

    session.execute = AsyncMock(side_effect=[insert_result, select_result])
    return session


@pytest.mark.asyncio
async def test_returns_existing_user() -> None:
    """If the user already exists, the SELECT returns it after the no-op upsert."""
    existing = User(telegram_user_id=111)
    existing.id = uuid.uuid4()
    session = _make_session(existing)

    result = await get_or_create_user(session, telegram_user_id=111)
    assert result is existing
    assert session.execute.await_count == 2
    # session.add should never be called — upsert is via execute
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_creates_user_when_not_found() -> None:
    """When upsert inserts a new row, the subsequent SELECT returns the new user."""
    new_user = User(telegram_user_id=999)
    new_user.id = uuid.uuid4()
    session = _make_session(new_user)

    result = await get_or_create_user(session, telegram_user_id=999)
    assert result is new_user
    assert session.execute.await_count == 2


@pytest.mark.asyncio
async def test_created_user_has_correct_telegram_id() -> None:
    """Returned user has the telegram_user_id that was passed in."""
    user = User(telegram_user_id=42)
    user.id = uuid.uuid4()
    session = _make_session(user)

    result = await get_or_create_user(session, telegram_user_id=42)
    assert result.telegram_user_id == 42
