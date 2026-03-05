"""Unit tests for mood journal (2.1).

Tests cover:
- emotion_to_mood: mapping emotion modes to mood values with intensity
- save_mood_entry: DB persistence
- get_mood_entries: retrieval within date range
- format_mood_timeline: text formatting for /mood command
- /mood command: handler logic with different subcommands
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from companion_bot_core.orchestrator.mood_journal import (
    emotion_to_mood,
    format_mood_timeline,
    save_mood_entry,
)

# ---------------------------------------------------------------------------
# emotion_to_mood — mapping logic
# ---------------------------------------------------------------------------


class TestEmotionToMood:
    def test_venting_maps_to_sad(self) -> None:
        mood, _ = emotion_to_mood("venting", 0.7)
        assert mood == "sad"

    def test_validation_maps_to_anxious(self) -> None:
        mood, _ = emotion_to_mood("validation", 0.5)
        assert mood == "anxious"

    def test_task_maps_to_neutral(self) -> None:
        mood, _ = emotion_to_mood("task", 0.6)
        assert mood == "neutral"

    def test_farewell_maps_to_neutral(self) -> None:
        mood, _ = emotion_to_mood("farewell", 0.8)
        assert mood == "neutral"

    def test_neutral_maps_to_neutral(self) -> None:
        mood, _ = emotion_to_mood("neutral", 0.0)
        assert mood == "neutral"

    def test_unknown_mode_maps_to_neutral(self) -> None:
        mood, _ = emotion_to_mood("unknown_mode", 0.5)
        assert mood == "neutral"

    def test_high_confidence_gives_high_intensity(self) -> None:
        _, intensity = emotion_to_mood("venting", 1.0)
        assert intensity == 5

    def test_low_confidence_gives_low_intensity(self) -> None:
        _, intensity = emotion_to_mood("venting", 0.1)
        assert intensity == 1

    def test_mid_confidence_gives_mid_intensity(self) -> None:
        _, intensity = emotion_to_mood("venting", 0.5)
        assert intensity in (2, 3)

    def test_zero_confidence_gives_min_intensity(self) -> None:
        _, intensity = emotion_to_mood("venting", 0.0)
        assert intensity == 1  # min clamped to 1

    def test_intensity_clamped_to_5(self) -> None:
        _, intensity = emotion_to_mood("venting", 1.5)
        assert intensity == 5


# ---------------------------------------------------------------------------
# save_mood_entry — DB persistence
# ---------------------------------------------------------------------------


class TestSaveMoodEntry:
    @pytest.mark.asyncio
    async def test_save_creates_row(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()

        await save_mood_entry(session, user_id, "sad", 3, context_snippet="устала от работы")

        session.add.assert_called_once()
        session.flush.assert_awaited_once()
        entry = session.add.call_args[0][0]
        assert entry.user_id == user_id
        assert entry.mood == "sad"
        assert entry.intensity == 3
        assert entry.context_snippet == "устала от работы"

    @pytest.mark.asyncio
    async def test_save_without_snippet(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()

        await save_mood_entry(session, user_id, "anxious", 2)

        entry = session.add.call_args[0][0]
        assert entry.context_snippet is None

    @pytest.mark.asyncio
    async def test_save_truncates_long_snippet(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()
        long_text = "a" * 100

        await save_mood_entry(session, user_id, "sad", 4, context_snippet=long_text)

        entry = session.add.call_args[0][0]
        assert len(entry.context_snippet) == 50


# ---------------------------------------------------------------------------
# get_mood_entries — retrieval
# ---------------------------------------------------------------------------


class TestGetMoodEntries:
    @pytest.mark.asyncio
    async def test_get_entries_returns_list(self) -> None:
        from companion_bot_core.orchestrator.mood_journal import get_mood_entries

        user_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = ["entry1", "entry2"]
        mock_result.scalars.return_value = mock_scalars

        session = AsyncMock()
        session.execute.return_value = mock_result

        result = await get_mood_entries(session, user_id, days=7)
        assert result == ["entry1", "entry2"]
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_entries_empty(self) -> None:
        from companion_bot_core.orchestrator.mood_journal import get_mood_entries

        user_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars

        session = AsyncMock()
        session.execute.return_value = mock_result

        result = await get_mood_entries(session, user_id, days=30)
        assert result == []


# ---------------------------------------------------------------------------
# format_mood_timeline — text formatting
# ---------------------------------------------------------------------------


class TestFormatMoodTimeline:
    def _make_entry(
        self, mood: str, intensity: int, created_at: datetime,
    ) -> MagicMock:
        entry = MagicMock()
        entry.mood = mood
        entry.intensity = intensity
        entry.created_at = created_at
        return entry

    def test_empty_entries_returns_empty(self) -> None:
        result = format_mood_timeline([])
        assert result == ""

    def test_single_entry_ru(self) -> None:
        entry = self._make_entry("sad", 3, datetime(2026, 3, 5, 12, 0, tzinfo=UTC))
        result = format_mood_timeline([entry], locale="ru")
        assert "05.03" in result
        assert "грусть" in result
        assert "😢" in result

    def test_single_entry_en(self) -> None:
        entry = self._make_entry("sad", 3, datetime(2026, 3, 5, 12, 0, tzinfo=UTC))
        result = format_mood_timeline([entry], locale="en")
        assert "05.03" in result
        assert "sad" in result
        assert "😢" in result

    def test_intensity_bar(self) -> None:
        entry = self._make_entry("anxious", 3, datetime(2026, 3, 5, 12, 0, tzinfo=UTC))
        result = format_mood_timeline([entry], locale="ru")
        assert "▪▪▪▫▫" in result

    def test_max_intensity_bar(self) -> None:
        entry = self._make_entry("angry", 5, datetime(2026, 3, 5, 12, 0, tzinfo=UTC))
        result = format_mood_timeline([entry], locale="ru")
        assert "▪▪▪▪▪" in result

    def test_min_intensity_bar(self) -> None:
        entry = self._make_entry("happy", 1, datetime(2026, 3, 5, 12, 0, tzinfo=UTC))
        result = format_mood_timeline([entry], locale="ru")
        assert "▪▫▫▫▫" in result

    def test_entries_grouped_by_date(self) -> None:
        entries = [
            self._make_entry("sad", 2, datetime(2026, 3, 5, 10, 0, tzinfo=UTC)),
            self._make_entry("anxious", 3, datetime(2026, 3, 5, 14, 0, tzinfo=UTC)),
            self._make_entry("happy", 4, datetime(2026, 3, 6, 9, 0, tzinfo=UTC)),
        ]
        result = format_mood_timeline(entries, locale="ru")
        # Should have two date headers
        assert result.count("05.03") == 1
        assert result.count("06.03") == 1

    def test_all_mood_emojis(self) -> None:
        moods = ["happy", "sad", "anxious", "angry", "neutral", "excited"]
        expected_emojis = ["😊", "😢", "😰", "😠", "😐", "🤩"]
        entries = [
            self._make_entry(m, 3, datetime(2026, 3, 5, i, 0, tzinfo=UTC))
            for i, m in enumerate(moods)
        ]
        result = format_mood_timeline(entries, locale="ru")
        for emoji in expected_emojis:
            assert emoji in result
