"""DB-backed conversation session tracker.

Maintains ``ConversationSession`` rows in the database.  A new session
starts when the gap since the user's last message exceeds
``SESSION_GAP_SECONDS`` (default 30 minutes).

Called from the orchestrator pipeline after inference, alongside the
Redis-based lightweight session counter.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select, text

from companion_bot_core.db.models import ConversationSession
from companion_bot_core.logging_config import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)

SESSION_GAP_SECONDS = 1800  # 30 minutes


async def track_session(
    db_session: AsyncSession,
    user_id: UUID,
    *,
    emotion_mode: str | None = None,
    is_farewell: bool = False,
) -> None:
    """Record a message in the user's current conversation session.

    - If no active session exists (or the last one ended > 30 min ago),
      creates a new ``ConversationSession``.
    - Otherwise, increments ``message_count`` and updates ``ended_at``.
    - Updates ``dominant_mood`` when *emotion_mode* is non-neutral.
    - Sets ``ended_with_farewell`` when *is_farewell* is ``True``.
    """
    now = datetime.now(tz=UTC)
    gap_threshold = now - timedelta(seconds=SESSION_GAP_SECONDS)

    # Transaction-scoped advisory lock on the user_id prevents two
    # concurrent first messages from both seeing no existing session and
    # creating duplicate rows.  The lock is released automatically when
    # the enclosing transaction commits or rolls back.
    lock_id = user_id.int & 0x7FFFFFFFFFFFFFFF  # fit into bigint
    await db_session.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": lock_id})

    # Find the most recent session for this user.
    # FOR UPDATE prevents concurrent messages from reading stale
    # message_count and losing increments.
    q = (
        select(ConversationSession)
        .where(ConversationSession.user_id == user_id)
        .order_by(ConversationSession.ended_at.desc())
        .limit(1)
        .with_for_update()
    )
    result = await db_session.execute(q)
    current = result.scalar_one_or_none()

    if (
        current is not None
        and current.ended_at >= gap_threshold
        and not current.ended_with_farewell
    ):
        # Continue existing session.
        current.message_count += 1
        current.ended_at = now
        if emotion_mode and emotion_mode != "neutral":
            current.dominant_mood = emotion_mode
        if is_farewell:
            current.ended_with_farewell = True
    else:
        # Start a new session.
        mood = emotion_mode if emotion_mode and emotion_mode != "neutral" else None
        new_session = ConversationSession(
            user_id=user_id,
            started_at=now,
            ended_at=now,
            message_count=1,
            dominant_mood=mood,
            ended_with_farewell=is_farewell,
        )
        db_session.add(new_session)

    await db_session.flush()
