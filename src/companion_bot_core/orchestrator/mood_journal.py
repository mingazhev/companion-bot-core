"""Mood journal: automatic mood tracking from emotion detection.

Maps emotion detector output to mood entries and provides query functions
for the ``/mood`` command.

Public surface:
    Mood                — literal type for the six mood values
    emotion_to_mood     — convert EmotionMode + confidence to (mood, intensity)
    save_mood_entry     — persist a mood entry to the database
    get_mood_entries    — retrieve mood entries for a user within a date range
    format_mood_timeline — format mood entries into a user-facing text timeline
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, Literal

from sqlalchemy import select

from companion_bot_core.db.models import MoodEntry
from companion_bot_core.logging_config import get_logger
from companion_bot_core.privacy.field_encryption import NOOP_ENCRYPTOR, FieldEncryptor

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)

Mood = Literal["happy", "sad", "anxious", "angry", "neutral", "excited"]

# Map EmotionMode → Mood.
# venting covers sadness/frustration; validation → anxious (seeking reassurance);
# task and farewell don't carry strong mood signals → neutral.
_EMOTION_TO_MOOD: Final[dict[str, Mood]] = {
    "venting": "sad",
    "validation": "anxious",
    "task": "neutral",
    "farewell": "neutral",
    "neutral": "neutral",
}


def emotion_to_mood(
    emotion_mode: str,
    confidence: float,
) -> tuple[Mood, int]:
    """Convert an emotion mode and confidence to a mood value and intensity.

    Returns:
        A ``(mood, intensity)`` tuple where intensity is 1-5.
    """
    mood = _EMOTION_TO_MOOD.get(emotion_mode, "neutral")
    # Map confidence (0.0-1.0) to intensity (1-5).
    intensity = max(1, min(5, round(confidence * 5)))
    return mood, intensity


async def save_mood_entry(
    db_session: AsyncSession,
    user_id: UUID,
    mood: Mood,
    intensity: int,
    context_snippet: str | None = None,
    encryptor: FieldEncryptor | None = None,
) -> None:
    """Persist a mood entry to the database."""
    enc = encryptor or NOOP_ENCRYPTOR
    snippet = context_snippet[:50] if context_snippet else None
    if snippet is not None:
        snippet = enc.encrypt(snippet)
    entry = MoodEntry(
        user_id=user_id,
        mood=mood,
        intensity=intensity,
        context_snippet=snippet,
    )
    db_session.add(entry)
    await db_session.flush()


async def get_mood_entries(
    db_session: AsyncSession,
    user_id: UUID,
    days: int = 7,
) -> list[MoodEntry]:
    """Retrieve mood entries for a user within the last *days* days."""
    since = datetime.now(tz=UTC) - timedelta(days=days)
    q = (
        select(MoodEntry)
        .where(MoodEntry.user_id == user_id)
        .where(MoodEntry.created_at >= since)
        .order_by(MoodEntry.created_at.asc())
    )
    result = await db_session.execute(q)
    return list(result.scalars().all())


# Mood → emoji for timeline display.
_MOOD_EMOJI: Final[dict[str, str]] = {
    "happy": "😊",
    "sad": "😢",
    "anxious": "😰",
    "angry": "😠",
    "neutral": "😐",
    "excited": "🤩",
}

# Mood → RU label.
_MOOD_LABEL_RU: Final[dict[str, str]] = {
    "happy": "радость",
    "sad": "грусть",
    "anxious": "тревога",
    "angry": "злость",
    "neutral": "спокойствие",
    "excited": "вдохновение",
}

# Mood → EN label.
_MOOD_LABEL_EN: Final[dict[str, str]] = {
    "happy": "happy",
    "sad": "sad",
    "anxious": "anxious",
    "angry": "angry",
    "neutral": "calm",
    "excited": "excited",
}


def format_mood_timeline(
    entries: list[MoodEntry],
    locale: str = "ru",
) -> str:
    """Format mood entries into a human-readable text timeline.

    Groups entries by date and shows mood emoji + label.
    """
    if not entries:
        return ""

    labels = _MOOD_LABEL_RU if locale.startswith("ru") else _MOOD_LABEL_EN
    lines: list[str] = []
    current_date: str | None = None

    for entry in entries:
        date_str = entry.created_at.strftime("%d.%m")
        emoji = _MOOD_EMOJI.get(entry.mood, "😐")
        label = labels.get(entry.mood, entry.mood)
        intensity_bar = "▪" * entry.intensity + "▫" * (5 - entry.intensity)

        if date_str != current_date:
            current_date = date_str
            lines.append(f"\n{date_str}:")

        lines.append(f"  {emoji} {label} [{intensity_bar}]")

    return "\n".join(lines).strip()
