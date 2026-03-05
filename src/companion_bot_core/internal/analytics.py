"""Session analytics queries for the internal HTTP service.

Provides functions that compute engagement metrics from the
``conversation_sessions`` table.  Used by the ``/internal/analytics/*``
route handlers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from companion_bot_core.db.models import ConversationSession

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


async def get_analytics_overview(
    session: AsyncSession,
    *,
    days: int = 7,
) -> dict[str, Any]:
    """Return aggregate engagement metrics over the last *days* days.

    Keys
    ----
    active_users : int
        Distinct users with at least one session in the window.
    total_sessions : int
        Total sessions started in the window.
    avg_session_messages : float
        Average messages per session.
    avg_session_duration_seconds : float
        Average session length in seconds (ended_at - started_at).
    farewell_rate : float
        Fraction of sessions that ended with a farewell.
    return_rate : float
        Fraction of active users who had more than one session.
    """
    since = datetime.now(tz=UTC) - timedelta(days=days)

    # --- active users ---
    q_users = select(
        func.count(func.distinct(ConversationSession.user_id))
    ).where(ConversationSession.started_at >= since)
    active_users: int = (await session.execute(q_users)).scalar_one()

    # --- total sessions ---
    q_total = select(func.count(ConversationSession.id)).where(
        ConversationSession.started_at >= since
    )
    total_sessions: int = (await session.execute(q_total)).scalar_one()

    # --- avg messages ---
    q_avg_msgs = select(
        func.coalesce(func.avg(ConversationSession.message_count), 0)
    ).where(ConversationSession.started_at >= since)
    avg_messages: float = float((await session.execute(q_avg_msgs)).scalar_one())

    # --- avg duration ---
    q_avg_dur = select(
        func.coalesce(
            func.avg(
                func.extract("epoch", ConversationSession.ended_at)
                - func.extract("epoch", ConversationSession.started_at)
            ),
            0,
        )
    ).where(ConversationSession.started_at >= since)
    avg_duration: float = float((await session.execute(q_avg_dur)).scalar_one())

    # --- farewell rate ---
    q_farewells = select(func.count(ConversationSession.id)).where(
        ConversationSession.started_at >= since,
        ConversationSession.ended_with_farewell.is_(True),
    )
    farewell_count: int = (await session.execute(q_farewells)).scalar_one()
    farewell_rate = farewell_count / total_sessions if total_sessions else 0.0

    # --- return rate ---
    # Subquery: users with session count > 1
    if active_users > 0:
        sub = (
            select(
                ConversationSession.user_id,
                func.count(ConversationSession.id).label("cnt"),
            )
            .where(ConversationSession.started_at >= since)
            .group_by(ConversationSession.user_id)
            .having(func.count(ConversationSession.id) > 1)
            .subquery()
        )
        q_returning = select(func.count()).select_from(sub)
        returning_users: int = (await session.execute(q_returning)).scalar_one()
        return_rate = returning_users / active_users
    else:
        return_rate = 0.0

    return {
        "period_days": days,
        "active_users": active_users,
        "total_sessions": total_sessions,
        "avg_session_messages": round(avg_messages, 2),
        "avg_session_duration_seconds": round(avg_duration, 2),
        "farewell_rate": round(farewell_rate, 4),
        "return_rate": round(return_rate, 4),
    }


async def get_user_analytics(
    session: AsyncSession,
    user_id: UUID,
    *,
    days: int = 30,
) -> dict[str, Any]:
    """Return per-user engagement profile over the last *days* days.

    Keys
    ----
    user_id : str
    total_sessions : int
    total_messages : int
    avg_session_messages : float
    avg_session_duration_seconds : float
    farewell_rate : float
    last_session_at : str | None
        ISO timestamp of the most recent session start.
    dominant_moods : dict[str, int]
        Mood distribution across sessions.
    """
    since = datetime.now(tz=UTC) - timedelta(days=days)

    q_sessions = select(ConversationSession).where(
        ConversationSession.user_id == user_id,
        ConversationSession.started_at >= since,
    ).order_by(ConversationSession.started_at.desc())

    rows = (await session.execute(q_sessions)).scalars().all()

    if not rows:
        return {
            "user_id": str(user_id),
            "total_sessions": 0,
            "total_messages": 0,
            "avg_session_messages": 0.0,
            "avg_session_duration_seconds": 0.0,
            "farewell_rate": 0.0,
            "last_session_at": None,
            "dominant_moods": {},
        }

    total_sessions = len(rows)
    total_messages = sum(r.message_count for r in rows)
    avg_messages = total_messages / total_sessions
    durations = [
        (r.ended_at - r.started_at).total_seconds() for r in rows
    ]
    avg_duration = sum(durations) / total_sessions
    farewell_count = sum(1 for r in rows if r.ended_with_farewell)
    farewell_rate = farewell_count / total_sessions

    mood_counts: dict[str, int] = {}
    for r in rows:
        if r.dominant_mood:
            mood_counts[r.dominant_mood] = mood_counts.get(r.dominant_mood, 0) + 1

    return {
        "user_id": str(user_id),
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "avg_session_messages": round(avg_messages, 2),
        "avg_session_duration_seconds": round(avg_duration, 2),
        "farewell_rate": round(farewell_rate, 4),
        "last_session_at": rows[0].started_at.isoformat(),
        "dominant_moods": mood_counts,
    }
