"""Habit tracking: creation, check-in, and streak management.

Detects habit creation intent and check-in via natural language,
manages streaks, and provides formatting for the /habits command.

Public surface:
    is_habit_create_request -- check if the message wants to create a habit
    extract_habit_title     -- extract habit title from creation message
    check_habit_match       -- match message against user's active habits
    checkin_habit           -- mark a habit as done, update streak
    calculate_streak        -- compute streak from last_checked_at
    create_habit            -- persist a new habit
    get_active_habits       -- list non-archived habits
    archive_habit           -- soft-delete a habit
    format_habits_list      -- format habits for /habits command
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

from companion_bot_core.db.models import Habit
from companion_bot_core.logging_config import get_logger
from companion_bot_core.signals import Signal, compile_signals, score_signals

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Habit creation signals
# ---------------------------------------------------------------------------

_HABIT_CREATE_THRESHOLD: Final[float] = 0.4

_HABIT_CREATE_SIGNALS: Final[list[Signal]] = compile_signals(
    [
        # Russian
        (r"\bхочу\s+привычку\b", 0.9),
        (r"\bхочу\s+каждый\s+день\b", 0.8),
        (r"\bхочу\s+каждую\s+неделю\b", 0.8),
        (r"\bновая\s+привычка\b", 0.9),
        (r"\bдобав\w*\s+привычку\b", 0.9),
        (r"\bзавес\w*\s+привычку\b", 0.9),
        (r"\bначн\w*\s+(каждый\s+день|ежедневно)\b", 0.7),
        (r"\bтрекать\b.{0,20}\bпривычк\w*\b", 0.8),
        (r"\bотслеживать\b.{0,20}\bпривычк\w*\b", 0.8),
        # English
        (r"\bnew\s+habit\b", 0.9),
        (r"\badd\s+(?:a\s+)?habit\b", 0.9),
        (r"\bcreate\s+(?:a\s+)?habit\b", 0.9),
        (r"\bstart\s+(?:a\s+)?habit\b", 0.8),
        (r"\bwant\s+to\b.{0,30}\bevery\s+day\b", 0.7),
        (r"\btrack\s+(?:a\s+)?habit\b", 0.8),
        (r"\bi\s+want\s+(?:a\s+)?habit\b", 0.8),
    ],
    dotall=True,
)

# Patterns to extract habit title from creation messages
_TITLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:хочу\s+каждый\s+день|хочу\s+каждую\s+неделю|начну\s+каждый\s+день)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:привычк[уа]|habit)\s*[:\-]?\s*(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:add|create|start|new)\s+(?:a\s+)?habit\s*[:\-]?\s*(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:добав\w+|завед\w+)\s+привычку\s*[:\-]?\s*(.+)",
        re.IGNORECASE,
    ),
]

# Words to strip from extracted titles
_STRIP_WORDS = re.compile(
    r"^(пожалуйста|please|давай|let'?s)\s+",
    re.IGNORECASE,
)

# Max active habits per user
MAX_ACTIVE_HABITS: Final[int] = 20
DEFAULT_LIMIT: Final[int] = 20


def is_habit_create_request(text: str) -> bool:
    """Return True if *text* looks like a request to create a habit."""
    return score_signals(text, _HABIT_CREATE_SIGNALS) >= _HABIT_CREATE_THRESHOLD


def extract_habit_title(text: str) -> str | None:
    """Extract a habit title from a creation message.

    Returns the cleaned title or None if extraction fails.
    """
    for pattern in _TITLE_PATTERNS:
        match = pattern.search(text)
        if match:
            title = match.group(1).strip().rstrip(".,!?;:")
            title = _STRIP_WORDS.sub("", title).strip()
            if title and len(title) <= 256:  # noqa: PLR2004
                return title
    return None


def _stem(word: str, min_len: int = 3) -> str:
    """Return a crude stem by truncating to the first *min_len* characters.

    For Russian, this approximates prefix-based stemming well enough for
    habit matching (e.g., "читать" → "чита", "читала" → "чита").
    """
    return word[:max(min_len, len(word) * 2 // 3)]


_QUESTION_RE = re.compile(
    r"\?\s*$"
    r"|^\s*(?:как|что|можно|когда|где|зачем|почему|кто|сколько|какой|какая|какие|какое)\b",
    re.IGNORECASE,
)


def check_habit_match(
    text: str,
    habits: list[Habit],
) -> Habit | None:
    """Check if *text* mentions any of the user's active habits.

    Uses word-boundary matching for the full title and prefix-based stem
    matching for individual words (handles conjugation like read/reading).
    Returns the first matched habit, or None.

    To prevent false positives, questions are skipped and multi-word
    habit titles require at least 2 matching words (a single shared stem
    like "клиент" is not enough to trigger a check-in).
    """
    if _QUESTION_RE.search(text):
        return None

    lower = text.lower()
    text_words = [w for w in lower.split() if len(w) >= 3]  # noqa: PLR2004
    for habit in habits:
        title_lower = habit.title.lower()
        title_words = [w for w in title_lower.split() if len(w) >= 3]  # noqa: PLR2004
        if not title_words:
            title_words = title_lower.split()
        # Word-boundary check for the full title to avoid matching substrings
        # (e.g., habit "run" should not match "returned").
        if re.search(rf"\b{re.escape(title_lower)}\b", lower):
            return habit
        # Prefix-based stem matching: check if any title word stem is a
        # prefix of a message word stem or vice versa.  This handles
        # conjugation (read→reading, читать→читала).
        matched_count = 0
        for tw in title_words:
            ts = _stem(tw)
            for mw in text_words:
                ms = _stem(mw)
                if ms.startswith(ts) or ts.startswith(ms):
                    matched_count += 1
                    break
        # Single-word titles: 1 match suffices.
        # Multi-word titles (2+): require at least 2 matching words to avoid
        # false positives from a single shared stem (e.g. "клиентов" matching
        # habit "делать 3 звонка потенциальным клиентам").
        min_matches = 1 if len(title_words) <= 1 else 2
        if matched_count >= min_matches:
            return habit
    return None


def calculate_streak(
    habit: Habit,
    now: datetime | None = None,
) -> int:
    """Return the current streak, accounting for missed days.

    If last_checked_at is more than the allowed gap (2 days for daily,
    14 days for weekly), streak resets to 0.
    """
    if habit.last_checked_at is None:
        return 0

    current = now or datetime.now(tz=UTC)
    gap = current - habit.last_checked_at

    max_gap = timedelta(days=14) if habit.frequency == "weekly" else timedelta(days=2)

    if gap > max_gap:
        return 0
    return habit.current_streak


async def create_habit(
    session: AsyncSession,
    user_id: UUID,
    title: str,
    frequency: str = "daily",
) -> Habit:
    """Create and persist a new habit."""
    habit = Habit(
        user_id=user_id,
        title=title,
        frequency=frequency,
    )
    session.add(habit)
    await session.flush()
    log.info("habit_created", user_id=str(user_id), title=title, frequency=frequency)
    return habit


async def checkin_habit(
    session: AsyncSession,
    habit: Habit,
    now: datetime | None = None,
) -> tuple[int, bool, bool]:
    """Mark a habit as done. Returns (new_streak, is_new_best, is_duplicate).

    When the habit was already checked in for the current period,
    ``is_duplicate`` is ``True`` and no DB changes are made.
    Streak resets silently if the gap since last check-in exceeds the
    frequency threshold.
    """
    current = now or datetime.now(tz=UTC)

    # Check if already checked in today (or this week for weekly)
    if habit.last_checked_at is not None:
        if habit.frequency == "weekly":
            same_period = (
                habit.last_checked_at.isocalendar()[:2] == current.isocalendar()[:2]
            )
        else:
            same_period = habit.last_checked_at.date() == current.date()
        if same_period:
            return habit.current_streak, False, True

    # Calculate effective streak considering gaps
    effective = calculate_streak(habit, current)
    new_streak = effective + 1

    habit.current_streak = new_streak
    habit.last_checked_at = current
    is_new_best = new_streak > habit.best_streak
    if is_new_best:
        habit.best_streak = new_streak

    await session.flush()
    log.info(
        "habit_checkin",
        habit_id=str(habit.id),
        title=habit.title,
        streak=new_streak,
        is_new_best=is_new_best,
    )
    return new_streak, is_new_best, False


async def get_active_habits(
    session: AsyncSession,
    user_id: UUID,
    *,
    limit: int = DEFAULT_LIMIT,
    for_update: bool = False,
) -> list[Habit]:
    """Return active (non-archived) habits for a user, newest first."""
    from sqlalchemy import select

    stmt = (
        select(Habit)
        .where(Habit.user_id == user_id, Habit.archived_at.is_(None))
        .order_by(Habit.created_at.desc())
        .limit(limit)
    )
    if for_update:
        stmt = stmt.with_for_update()
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def archive_habit(
    session: AsyncSession,
    habit: Habit,
) -> None:
    """Soft-delete a habit by setting archived_at."""
    habit.archived_at = datetime.now(tz=UTC)
    await session.flush()
    log.info("habit_archived", habit_id=str(habit.id), title=habit.title)


def format_habits_list(
    habits: list[Habit],
    locale: str = "ru",
) -> str:
    """Format a list of habits for display in /habits command."""
    if not habits:
        return ""

    now = datetime.now(tz=UTC)
    lines: list[str] = []
    for i, habit in enumerate(habits, start=1):
        streak = calculate_streak(habit, now)
        freq = habit.frequency
        freq_label = {"daily": "ежедн." if locale == "ru" else "daily",
                      "weekly": "еженед." if locale == "ru" else "weekly"}.get(freq, freq)

        # Streak bar visualization
        bar = _streak_bar(streak)

        # Check if done today
        if habit.last_checked_at is not None:
            if habit.frequency == "weekly":
                done_today = (
                    habit.last_checked_at.isocalendar()[:2] == now.isocalendar()[:2]
                )
            else:
                done_today = habit.last_checked_at.date() == now.date()
        else:
            done_today = False

        check = "v" if done_today else " "
        best_label = "рекорд" if locale == "ru" else "best"

        entry = (
            f"{i}. [{check}] {habit.title} ({freq_label})\n"
            f"   {bar} {streak} | {best_label}: {habit.best_streak}"
        )
        lines.append(entry)

    return "\n\n".join(lines)


def _streak_bar(streak: int) -> str:
    """Return a simple text-based streak visualization."""
    filled = min(streak, 7)
    return "|" * filled + "." * (7 - filled)
