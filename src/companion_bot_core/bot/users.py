"""User get-or-create helper.

Provides an atomic upsert pattern: SELECT first, INSERT only when the user
does not exist yet.  The flush() call after insert assigns the DB-generated
fields (server_default) so the returned object is fully usable within the
same transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from companion_bot_core.db.models import User

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_or_create_user(session: AsyncSession, telegram_user_id: int) -> User:
    """Return the existing :class:`User` row for *telegram_user_id*, creating it if absent.

    Uses an upsert (INSERT … ON CONFLICT DO NOTHING) to avoid a race condition
    when two concurrent requests arrive for the same new user.

    The caller is responsible for committing the enclosing transaction.
    """
    stmt = (
        insert(User)
        .values(telegram_user_id=telegram_user_id, locale="ru")
        .on_conflict_do_nothing(index_elements=["telegram_user_id"])
    )
    await session.execute(stmt)
    result = await session.execute(
        select(User).where(User.telegram_user_id == telegram_user_id)
    )
    return result.scalar_one()
