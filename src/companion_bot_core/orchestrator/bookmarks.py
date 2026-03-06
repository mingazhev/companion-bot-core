"""Bookmark detection and persistence.

Detects when a user wants to save a conversation moment and persists
the previous user message + bot response pair as a bookmark.

Public surface:
    is_bookmark_request  -- check if the message is a bookmark request
    save_bookmark        -- persist a bookmark to the database
    get_bookmarks        -- retrieve user's bookmarks
    search_bookmarks     -- search bookmarks by text query
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from companion_bot_core.db.models import Bookmark
from companion_bot_core.logging_config import get_logger
from companion_bot_core.signals import Signal, compile_signals, score_signals

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)

# Minimum score to consider the message a bookmark request.
_BOOKMARK_THRESHOLD: Final[float] = 0.4

_BOOKMARK_SIGNALS: Final[list[Signal]] = compile_signals(
    [
        # Russian
        (r"\bзапомни\s+(это|момент)\b", 0.9),
        (r"\bсохрани\s+(это|момент|разговор)\b", 0.9),
        (r"\bэто\s+важно\b", 0.8),
        (r"\bзакладк[уа]\b", 0.7),
        (r"\bне\s+забудь\s+(это|этот\s+момент)\b", 0.8),
        (r"\bзапиши\s+(это|себе)\b", 0.7),
        # English
        (r"\bremember\s+this\b", 0.9),
        (r"\bsave\s+this\b", 0.9),
        (r"\bthis\s+is\s+important\b", 0.8),
        (r"\bbookmark\s+(this|it)\b", 0.9),
        (r"\bkeep\s+this\b", 0.7),
        (r"\bnote\s+this\b", 0.7),
        (r"\bdon'?t\s+forget\s+this\b", 0.8),
    ],
    dotall=True,
)

# Maximum number of bookmarks returned by get_bookmarks.
DEFAULT_LIMIT: Final[int] = 20


def is_bookmark_request(text: str) -> bool:
    """Return True if *text* looks like a bookmark request."""
    return score_signals(text, _BOOKMARK_SIGNALS) >= _BOOKMARK_THRESHOLD


async def save_bookmark(
    session: AsyncSession,
    user_id: UUID,
    user_message: str,
    bot_response: str,
    *,
    tag: str | None = None,
) -> Bookmark:
    """Persist a new bookmark and return the created row."""
    bookmark = Bookmark(
        user_id=user_id,
        user_message=user_message,
        bot_response=bot_response,
        tag=tag,
    )
    session.add(bookmark)
    await session.flush()
    log.info(
        "bookmark_saved",
        user_id=str(user_id),
        tag=tag,
    )
    return bookmark


async def get_bookmarks(
    session: AsyncSession,
    user_id: UUID,
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[Bookmark]:
    """Return the most recent bookmarks for a user, newest first."""
    from sqlalchemy import select

    stmt = (
        select(Bookmark)
        .where(Bookmark.user_id == user_id)
        .order_by(Bookmark.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def search_bookmarks(
    session: AsyncSession,
    user_id: UUID,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[Bookmark]:
    """Search user's bookmarks by text content (case-insensitive LIKE)."""
    from sqlalchemy import or_, select

    escaped = query.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    pattern = f"%{escaped}%"
    stmt = (
        select(Bookmark)
        .where(
            Bookmark.user_id == user_id,
            or_(
                Bookmark.user_message.ilike(pattern, escape="\\"),
                Bookmark.bot_response.ilike(pattern, escape="\\"),
                Bookmark.tag.ilike(pattern, escape="\\"),
            ),
        )
        .order_by(Bookmark.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
