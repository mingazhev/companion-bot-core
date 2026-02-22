"""User get-or-create helper.

Provides an atomic upsert pattern: SELECT first, INSERT only when the user
does not exist yet.  The flush() call after insert assigns the DB-generated
fields (server_default) so the returned object is fully usable within the
same transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from tdbot.db.models import User

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_or_create_user(session: AsyncSession, telegram_user_id: int) -> User:
    """Return the existing :class:`User` row for *telegram_user_id*, creating it if absent.

    The caller is responsible for committing the enclosing transaction.
    """
    result = await session.execute(
        select(User).where(User.telegram_user_id == telegram_user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        user = User(telegram_user_id=telegram_user_id)
        session.add(user)
        await session.flush()
    return user
