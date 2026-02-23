"""TTL sweeper: batch-delete expired conversation_messages rows.

The ``conversation_messages`` table has a ``ttl_expires_at`` column.  Rows
whose ``ttl_expires_at`` timestamp is in the past are eligible for deletion.
This module provides the sweeper function that performs that deletion.

Intended use
------------
Call ``sweep_expired_messages`` from a periodic background task (e.g., an
asyncio task that wakes up on a configurable cadence) or from the ``jobs``
queue handler when a ``ttl_sweep`` job is dequeued::

    from tdbot.privacy.ttl_sweeper import sweep_expired_messages

    deleted = await sweep_expired_messages(session)
    log.info("ttl_sweep_done", deleted=deleted)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import delete, func

from tdbot.db.models import ConversationMessage

if TYPE_CHECKING:
    from sqlalchemy.engine import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession


async def sweep_expired_messages(session: AsyncSession) -> int:
    """Delete all ``conversation_messages`` rows whose TTL has expired.

    A row is eligible when ``ttl_expires_at IS NOT NULL`` and
    ``ttl_expires_at < now()``.

    Args:
        session: An active :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
                 The caller is responsible for committing the transaction.

    Returns:
        The number of rows deleted.
    """
    result = await session.execute(
        delete(ConversationMessage)
        .where(ConversationMessage.ttl_expires_at.isnot(None))
        .where(ConversationMessage.ttl_expires_at < func.now())
    )
    return cast("CursorResult[Any]", result).rowcount
