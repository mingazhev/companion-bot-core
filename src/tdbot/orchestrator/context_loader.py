"""Context assembly for the conversation orchestrator.

Provides helpers to load recent conversation history from PostgreSQL and to
build the :class:`~tdbot.inference.schemas.UserContext` object that is passed
to the inference adapter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from sqlalchemy import select

from tdbot.db.models import ConversationMessage
from tdbot.inference.schemas import ChatMessage, UserContext
from tdbot.prompt.merge_builder import build_system_prompt
from tdbot.prompt.schemas import PromptComponents, SnapshotRecord

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from tdbot.prompt.snapshot_store import SnapshotStore

_DEFAULT_SYSTEM_TEMPLATE = "You are a helpful, friendly companion."
_DEFAULT_MAX_TOKENS = 1024
# Type alias for valid chat message roles — used to satisfy ChatMessage.role narrowing
_MessageRole = Literal["system", "user", "assistant"]


async def load_recent_messages(
    session: AsyncSession,
    user_id: UUID,
    limit: int = 20,
) -> list[ChatMessage]:
    """Return the most recent non-expired conversation messages for *user_id*.

    Messages are ordered **oldest first** so they can be fed directly into the
    message list as conversation history.

    Args:
        session:  Active database session.
        user_id:  Internal (UUID) user identifier.
        limit:    Maximum number of messages to load (default 20).

    Returns:
        List of :class:`~tdbot.inference.schemas.ChatMessage` in chronological
        order (oldest first).
    """
    now = datetime.now(tz=UTC)
    stmt = (
        select(ConversationMessage)
        .where(ConversationMessage.user_id == user_id)
        .where(
            (ConversationMessage.ttl_expires_at.is_(None))
            | (ConversationMessage.ttl_expires_at > now)
        )
        .order_by(ConversationMessage.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())

    # Reverse to chronological (oldest first) for the inference message list
    return [
        ChatMessage(role=cast("_MessageRole", row.role), content=row.content)
        for row in reversed(rows)
    ]


async def load_user_context(
    session: AsyncSession,
    snapshot_store: SnapshotStore,
    user_id: UUID,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> UserContext:
    """Build a :class:`~tdbot.inference.schemas.UserContext` for *user_id*.

    Steps:
    1. Load the active prompt snapshot from the store; fall back to a minimal
       default if no snapshot exists yet.
    2. Load the recent non-expired message history from the database.
    3. Assemble and return :class:`~tdbot.inference.schemas.UserContext`.

    Args:
        session:        Active database session (for message history).
        snapshot_store: Snapshot store (for active prompt snapshot).
        user_id:        Internal (UUID) user identifier.
        max_tokens:     Maximum completion tokens passed to the model.

    Returns:
        A populated :class:`~tdbot.inference.schemas.UserContext`.
    """
    snapshot = await snapshot_store.get_active(user_id)

    if snapshot is not None:
        system_prompt = snapshot.system_prompt
    else:
        components = PromptComponents(base_system_template=_DEFAULT_SYSTEM_TEMPLATE)
        system_prompt = build_system_prompt(components)
        # Persist an initial snapshot so the refinement worker has a base to
        # build on.  Without this, the worker always skips new users.
        version = await snapshot_store.next_version(user_id)
        initial = SnapshotRecord(
            user_id=user_id,
            version=version,
            system_prompt=system_prompt,
            source="initial",
        )
        await snapshot_store.save(initial)
        await snapshot_store.set_active(user_id, initial.id)

    history = await load_recent_messages(session, user_id, limit=20)

    return UserContext(
        user_id=str(user_id),
        system_prompt=system_prompt,
        conversation_history=history,
        max_tokens=max_tokens,
    )
