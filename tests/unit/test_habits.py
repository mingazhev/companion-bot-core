"""Unit tests for Goals & Habits (2.3).

Tests cover:
- habits.is_habit_create_request: intent detection for RU and EN triggers
- habits.extract_habit_title: title extraction from creation messages
- habits.check_habit_match: matching messages against existing habits
- habits.calculate_streak: streak calculation with gap handling
- habits.create_habit / checkin_habit: DB persistence and streak updates
- habits.get_active_habits: retrieval of non-archived habits
- habits.format_habits_list: formatting for /habits command
- Metric objects: sanity checks
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from companion_bot_core.orchestrator.habits import (
    calculate_streak,
    check_habit_match,
    checkin_habit,
    create_habit,
    extract_habit_title,
    format_habits_list,
    is_habit_create_request,
)

# ---------------------------------------------------------------------------
# is_habit_create_request — intent detection
# ---------------------------------------------------------------------------


class TestIsHabitCreateRequest:
    def test_hochu_privychku(self) -> None:
        assert is_habit_create_request("хочу привычку") is True

    def test_hochu_kazhdyj_den(self) -> None:
        assert is_habit_create_request("хочу каждый день читать") is True

    def test_hochu_kazhduju_nedelyu(self) -> None:
        assert is_habit_create_request("хочу каждую неделю бегать") is True

    def test_novaya_privychka(self) -> None:
        assert is_habit_create_request("новая привычка") is True

    def test_dobavit_privychku(self) -> None:
        assert is_habit_create_request("добавить привычку пить воду") is True

    def test_zavesti_privychku(self) -> None:
        assert is_habit_create_request("завести привычку медитировать") is True

    def test_new_habit_en(self) -> None:
        assert is_habit_create_request("new habit") is True

    def test_add_habit_en(self) -> None:
        assert is_habit_create_request("add a habit") is True

    def test_create_habit_en(self) -> None:
        assert is_habit_create_request("create a habit to read") is True

    def test_start_habit_en(self) -> None:
        assert is_habit_create_request("start a habit") is True

    def test_want_habit_en(self) -> None:
        assert is_habit_create_request("I want a habit") is True

    def test_track_habit_en(self) -> None:
        assert is_habit_create_request("track a habit") is True

    def test_case_insensitive(self) -> None:
        assert is_habit_create_request("ХОЧУ ПРИВЫЧКУ") is True
        assert is_habit_create_request("New Habit") is True

    def test_normal_message_not_detected(self) -> None:
        assert is_habit_create_request("привет, как дела?") is False

    def test_empty_string(self) -> None:
        assert is_habit_create_request("") is False

    def test_unrelated_message_not_detected(self) -> None:
        assert is_habit_create_request("я сегодня читала книгу") is False


# ---------------------------------------------------------------------------
# extract_habit_title — title extraction
# ---------------------------------------------------------------------------


class TestExtractHabitTitle:
    def test_extract_from_kazhdyj_den(self) -> None:
        result = extract_habit_title("хочу каждый день читать")
        assert result == "читать"

    def test_extract_from_privychku(self) -> None:
        result = extract_habit_title("добавить привычку пить воду")
        assert result == "пить воду"

    def test_extract_from_new_habit_en(self) -> None:
        result = extract_habit_title("new habit: read books")
        assert result == "read books"

    def test_extract_from_add_habit_en(self) -> None:
        result = extract_habit_title("add a habit to exercise")
        assert result == "to exercise"

    def test_extract_from_create_habit_en(self) -> None:
        result = extract_habit_title("create habit meditate")
        assert result == "meditate"

    def test_strips_pleasantries(self) -> None:
        result = extract_habit_title("привычка: пожалуйста пить воду")
        assert result == "пить воду"

    def test_strips_trailing_punctuation(self) -> None:
        result = extract_habit_title("new habit: read.")
        assert result == "read"

    def test_returns_none_for_empty(self) -> None:
        result = extract_habit_title("привет мир")
        assert result is None

    def test_returns_none_for_too_long(self) -> None:
        result = extract_habit_title(f"привычка: {'x' * 300}")
        assert result is None


# ---------------------------------------------------------------------------
# check_habit_match — matching against user habits
# ---------------------------------------------------------------------------


class TestCheckHabitMatch:
    def _make_habit(self, title: str) -> MagicMock:
        h = MagicMock()
        h.title = title
        return h

    def test_matches_habit_keyword(self) -> None:
        habits = [self._make_habit("читать"), self._make_habit("бегать")]
        assert check_habit_match("сегодня читала", habits) is habits[0]

    def test_matches_english_habit(self) -> None:
        habits = [self._make_habit("read"), self._make_habit("exercise")]
        assert check_habit_match("I read today", habits) is habits[0]

    def test_no_match(self) -> None:
        habits = [self._make_habit("читать")]
        assert check_habit_match("привет, как дела?", habits) is None

    def test_empty_habits_list(self) -> None:
        assert check_habit_match("сегодня читала", []) is None

    def test_case_insensitive(self) -> None:
        habits = [self._make_habit("Читать")]
        assert check_habit_match("ЧИТАЛА СЕГОДНЯ", habits) is habits[0]

    def test_first_match_wins(self) -> None:
        habits = [self._make_habit("read"), self._make_habit("reading")]
        assert check_habit_match("I was reading", habits) is habits[0]

    def test_short_title_words(self) -> None:
        habits = [self._make_habit("go")]
        assert check_habit_match("I need to go", habits) is habits[0]

    def test_question_skipped(self) -> None:
        """Questions should not trigger check-in."""
        habits = [self._make_habit("звонить клиентам")]
        assert check_habit_match("как удержать первых клиентов?", habits) is None

    def test_question_word_at_start_skipped(self) -> None:
        habits = [self._make_habit("читать")]
        assert check_habit_match("что читать дальше", habits) is None

    def test_multi_word_title_single_stem_no_match(self) -> None:
        """A single shared stem should not match a multi-word habit title."""
        habits = [self._make_habit("делать звонки потенциальным клиентам")]
        assert check_habit_match("ещё момент -- как удержать первых клиентов", habits) is None

    def test_multi_word_title_two_stems_match(self) -> None:
        """Two matching stems should be enough for a multi-word title."""
        habits = [self._make_habit("читать документацию")]
        assert check_habit_match("сегодня читала документацию", habits) is habits[0]


# ---------------------------------------------------------------------------
# calculate_streak — streak calculation
# ---------------------------------------------------------------------------


class TestCalculateStreak:
    def _make_habit(
        self,
        current_streak: int = 5,
        last_checked_at: datetime | None = None,
        frequency: str = "daily",
    ) -> MagicMock:
        h = MagicMock()
        h.current_streak = current_streak
        h.last_checked_at = last_checked_at
        h.frequency = frequency
        return h

    def test_returns_zero_if_never_checked(self) -> None:
        habit = self._make_habit(current_streak=0, last_checked_at=None)
        assert calculate_streak(habit) == 0

    def test_returns_streak_if_checked_recently(self) -> None:
        now = datetime.now(tz=UTC)
        habit = self._make_habit(
            current_streak=5,
            last_checked_at=now - timedelta(hours=12),
        )
        assert calculate_streak(habit, now) == 5

    def test_returns_zero_if_daily_gap_exceeded(self) -> None:
        now = datetime.now(tz=UTC)
        habit = self._make_habit(
            current_streak=5,
            last_checked_at=now - timedelta(days=3),
        )
        assert calculate_streak(habit, now) == 0

    def test_daily_boundary_day(self) -> None:
        now = datetime.now(tz=UTC)
        habit = self._make_habit(
            current_streak=3,
            last_checked_at=now - timedelta(days=1, hours=23),
        )
        assert calculate_streak(habit, now) == 3

    def test_daily_exactly_two_days(self) -> None:
        now = datetime.now(tz=UTC)
        habit = self._make_habit(
            current_streak=3,
            last_checked_at=now - timedelta(days=2),
        )
        assert calculate_streak(habit, now) == 3

    def test_daily_just_over_two_days(self) -> None:
        now = datetime.now(tz=UTC)
        habit = self._make_habit(
            current_streak=3,
            last_checked_at=now - timedelta(days=2, seconds=1),
        )
        assert calculate_streak(habit, now) == 0

    def test_weekly_returns_streak_within_threshold(self) -> None:
        now = datetime.now(tz=UTC)
        habit = self._make_habit(
            current_streak=4,
            last_checked_at=now - timedelta(days=10),
            frequency="weekly",
        )
        assert calculate_streak(habit, now) == 4

    def test_weekly_resets_after_14_days(self) -> None:
        now = datetime.now(tz=UTC)
        habit = self._make_habit(
            current_streak=4,
            last_checked_at=now - timedelta(days=15),
            frequency="weekly",
        )
        assert calculate_streak(habit, now) == 0


# ---------------------------------------------------------------------------
# create_habit — DB persistence
# ---------------------------------------------------------------------------


class TestCreateHabit:
    @pytest.mark.asyncio
    async def test_create_habit_persists(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()

        habit = await create_habit(session, user_id, "read", "daily")

        session.add.assert_called_once()
        session.flush.assert_awaited_once()
        assert habit.user_id == user_id
        assert habit.title == "read"
        assert habit.frequency == "daily"

    @pytest.mark.asyncio
    async def test_create_habit_weekly(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()

        habit = await create_habit(session, user_id, "run", "weekly")

        assert habit.frequency == "weekly"


# ---------------------------------------------------------------------------
# checkin_habit — check-in and streak updates
# ---------------------------------------------------------------------------


class TestCheckinHabit:
    @pytest.mark.asyncio
    async def test_first_checkin_sets_streak_to_one(self) -> None:
        session = AsyncMock()
        now = datetime.now(tz=UTC)
        habit = MagicMock()
        habit.id = uuid.uuid4()
        habit.title = "read"
        habit.frequency = "daily"
        habit.current_streak = 0
        habit.best_streak = 0
        habit.last_checked_at = None

        streak, is_new_best, is_dup = await checkin_habit(session, habit, now)

        assert streak == 1
        assert is_new_best is True
        assert is_dup is False
        assert habit.current_streak == 1
        assert habit.best_streak == 1
        assert habit.last_checked_at == now

    @pytest.mark.asyncio
    async def test_consecutive_checkin_increments_streak(self) -> None:
        session = AsyncMock()
        # Use a fixed time (noon) so that subtracting 20 hours always
        # falls on the previous calendar day, avoiding same-period dedup.
        now = datetime(2026, 3, 7, 12, 0, 0, tzinfo=UTC)
        habit = MagicMock()
        habit.id = uuid.uuid4()
        habit.title = "read"
        habit.frequency = "daily"
        habit.current_streak = 3
        habit.best_streak = 5
        habit.last_checked_at = now - timedelta(hours=20)

        streak, is_new_best, is_dup = await checkin_habit(session, habit, now)

        assert streak == 4
        assert is_new_best is False
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_gap_resets_streak(self) -> None:
        session = AsyncMock()
        now = datetime.now(tz=UTC)
        habit = MagicMock()
        habit.id = uuid.uuid4()
        habit.title = "read"
        habit.frequency = "daily"
        habit.current_streak = 10
        habit.best_streak = 15
        habit.last_checked_at = now - timedelta(days=5)

        streak, is_new_best, is_dup = await checkin_habit(session, habit, now)

        # Streak reset to 0 due to gap, then +1
        assert streak == 1
        assert is_new_best is False
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_same_day_no_double_checkin(self) -> None:
        session = AsyncMock()
        now = datetime.now(tz=UTC)
        habit = MagicMock()
        habit.id = uuid.uuid4()
        habit.title = "read"
        habit.frequency = "daily"
        habit.current_streak = 5
        habit.best_streak = 5
        habit.last_checked_at = now.replace(hour=8, minute=0, second=0)

        streak, is_new_best, is_dup = await checkin_habit(session, habit, now)

        # Should return existing streak without incrementing
        assert streak == 5
        assert is_new_best is False
        assert is_dup is True
        session.flush.assert_not_awaited()


# ---------------------------------------------------------------------------
# get_active_habits — retrieval
# ---------------------------------------------------------------------------


class TestGetActiveHabits:
    @pytest.mark.asyncio
    async def test_returns_habits_list(self) -> None:
        from companion_bot_core.orchestrator.habits import get_active_habits

        user_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = ["habit1", "habit2"]
        mock_result.scalars.return_value = mock_scalars

        session = AsyncMock()
        session.execute.return_value = mock_result

        result = await get_active_habits(session, user_id)
        assert result == ["habit1", "habit2"]
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_empty_list(self) -> None:
        from companion_bot_core.orchestrator.habits import get_active_habits

        user_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars

        session = AsyncMock()
        session.execute.return_value = mock_result

        result = await get_active_habits(session, user_id)
        assert result == []


# ---------------------------------------------------------------------------
# format_habits_list — formatting for /habits command
# ---------------------------------------------------------------------------


class TestFormatHabitsList:
    def _make_habit(
        self,
        title: str = "read",
        frequency: str = "daily",
        current_streak: int = 3,
        best_streak: int = 7,
        last_checked_at: datetime | None = None,
    ) -> MagicMock:
        h = MagicMock()
        h.title = title
        h.frequency = frequency
        h.current_streak = current_streak
        h.best_streak = best_streak
        h.last_checked_at = last_checked_at
        return h

    def test_empty_list(self) -> None:
        assert format_habits_list([]) == ""

    def test_single_habit_ru(self) -> None:
        habit = self._make_habit(title="читать", current_streak=3, best_streak=5)
        result = format_habits_list([habit], locale="ru")
        assert "читать" in result
        assert "ежедн." in result
        assert "рекорд: 5" in result

    def test_single_habit_en(self) -> None:
        habit = self._make_habit(title="read", current_streak=3, best_streak=5)
        result = format_habits_list([habit], locale="en")
        assert "read" in result
        assert "daily" in result
        assert "best: 5" in result

    def test_checked_today_shows_checkmark(self) -> None:
        now = datetime.now(tz=UTC)
        habit = self._make_habit(
            title="read",
            current_streak=3,
            last_checked_at=now,
        )
        result = format_habits_list([habit], locale="en")
        assert "[v]" in result

    def test_not_checked_today(self) -> None:
        yesterday = datetime.now(tz=UTC) - timedelta(days=1)
        habit = self._make_habit(
            title="read",
            current_streak=3,
            last_checked_at=yesterday,
        )
        result = format_habits_list([habit], locale="en")
        assert "[ ]" in result

    def test_multiple_habits(self) -> None:
        habits = [
            self._make_habit(title="read"),
            self._make_habit(title="exercise"),
            self._make_habit(title="meditate"),
        ]
        result = format_habits_list(habits, locale="en")
        assert "1." in result
        assert "2." in result
        assert "3." in result

    def test_weekly_habit_label(self) -> None:
        habit = self._make_habit(title="run", frequency="weekly")
        result = format_habits_list([habit], locale="en")
        assert "weekly" in result

    def test_streak_bar_visualization(self) -> None:
        now = datetime.now(tz=UTC)
        habit = self._make_habit(
            title="read",
            current_streak=3,
            last_checked_at=now - timedelta(hours=12),
        )
        result = format_habits_list([habit], locale="en")
        # 3 filled + 4 empty = |||....
        assert "|||...." in result


# ---------------------------------------------------------------------------
# Metric objects — sanity checks
# ---------------------------------------------------------------------------


class TestHabitMetrics:
    def test_habit_created_metric_exists(self) -> None:
        from companion_bot_core.metrics import HABIT_CREATED

        HABIT_CREATED.inc()

    def test_habit_checkin_metric_exists(self) -> None:
        from companion_bot_core.metrics import HABIT_CHECKIN

        HABIT_CHECKIN.inc()
