"""Unit tests for proactive messaging (2.2).

Tests cover:
- Warm return: build_warm_return_hint threshold logic
- Check-in scheduling: compute_next_fire, is_in_quiet_hours
- Check-in Redis operations: schedule, unschedule, get_due, mark_sent
- Scheduler: timezone parsing, process flow
- /checkin command: on/off/quiet/status flows
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, time, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from companion_bot_core.proactive.warm_return import (
    WARM_RETURN_GAP_SECONDS,
    build_warm_return_hint,
)

# ---------------------------------------------------------------------------
# Warm return — build_warm_return_hint
# ---------------------------------------------------------------------------


class TestBuildWarmReturnHint:
    def test_returns_empty_for_short_gap(self) -> None:
        result = build_warm_return_hint(3600, "ru")
        assert result == ""

    def test_returns_empty_for_zero_gap(self) -> None:
        result = build_warm_return_hint(0, "ru")
        assert result == ""

    def test_returns_empty_for_gap_just_below_threshold(self) -> None:
        result = build_warm_return_hint(WARM_RETURN_GAP_SECONDS - 1, "en")
        assert result == ""

    def test_returns_hint_at_exact_threshold(self) -> None:
        result = build_warm_return_hint(WARM_RETURN_GAP_SECONDS, "ru")
        assert result != ""
        assert "Давно не общались" in result

    def test_returns_hint_for_large_gap(self) -> None:
        result = build_warm_return_hint(WARM_RETURN_GAP_SECONDS * 10, "ru")
        assert result != ""

    def test_hint_locale_ru(self) -> None:
        result = build_warm_return_hint(WARM_RETURN_GAP_SECONDS, "ru")
        assert "Давно не общались" in result

    def test_hint_locale_en(self) -> None:
        result = build_warm_return_hint(WARM_RETURN_GAP_SECONDS, "en")
        assert "been a while" in result

    def test_hint_default_locale(self) -> None:
        result = build_warm_return_hint(WARM_RETURN_GAP_SECONDS, None)
        # Default locale is "ru"
        assert "Давно не общались" in result


# ---------------------------------------------------------------------------
# Check-in — is_in_quiet_hours
# ---------------------------------------------------------------------------


class TestIsInQuietHours:
    def test_no_quiet_hours_returns_false(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        assert is_in_quiet_hours(time(12, 0), None, None) is False

    def test_start_none_returns_false(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        assert is_in_quiet_hours(time(12, 0), None, time(8, 0)) is False

    def test_end_none_returns_false(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        assert is_in_quiet_hours(time(12, 0), time(22, 0), None) is False

    def test_same_day_range_inside(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        # Quiet 08:00-12:00, current 10:00 — inside
        assert is_in_quiet_hours(time(10, 0), time(8, 0), time(12, 0)) is True

    def test_same_day_range_outside(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        # Quiet 08:00-12:00, current 14:00 — outside
        assert is_in_quiet_hours(time(14, 0), time(8, 0), time(12, 0)) is False

    def test_midnight_wrap_inside_evening(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        # Quiet 22:00-08:00, current 23:00 — inside
        assert is_in_quiet_hours(time(23, 0), time(22, 0), time(8, 0)) is True

    def test_midnight_wrap_inside_morning(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        # Quiet 22:00-08:00, current 06:00 — inside
        assert is_in_quiet_hours(time(6, 0), time(22, 0), time(8, 0)) is True

    def test_midnight_wrap_outside(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        # Quiet 22:00-08:00, current 12:00 — outside
        assert is_in_quiet_hours(time(12, 0), time(22, 0), time(8, 0)) is False

    def test_boundary_start(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        # Quiet 22:00-08:00, current exactly 22:00 — inside
        assert is_in_quiet_hours(time(22, 0), time(22, 0), time(8, 0)) is True

    def test_boundary_end(self) -> None:
        from companion_bot_core.proactive.checkin import is_in_quiet_hours

        # Quiet 22:00-08:00, current exactly 08:00 — inside
        assert is_in_quiet_hours(time(8, 0), time(22, 0), time(8, 0)) is True


# ---------------------------------------------------------------------------
# Check-in — compute_next_fire
# ---------------------------------------------------------------------------


class TestComputeNextFire:
    def test_future_time_today(self) -> None:
        from companion_bot_core.proactive.checkin import compute_next_fire

        # If we ask for 23:59 and it's before that, should be today
        tz = UTC
        result = compute_next_fire(time(23, 59), tz)
        now = datetime.now(tz=tz)
        dt = datetime.fromtimestamp(result, tz=tz)
        # Should be today or tomorrow
        assert dt >= now
        assert dt.hour == 23
        assert dt.minute == 59

    def test_past_time_schedules_tomorrow(self) -> None:
        from companion_bot_core.proactive.checkin import compute_next_fire

        tz = UTC
        now = datetime.now(tz=tz)
        # Use a time that's definitely in the past (00:00 is past if it's not midnight)
        if now.hour > 0 or now.minute > 0:
            result = compute_next_fire(time(0, 0), tz)
            dt = datetime.fromtimestamp(result, tz=tz)
            assert dt.date() == now.date() + timedelta(days=1)

    def test_respects_timezone_offset(self) -> None:
        from companion_bot_core.proactive.checkin import compute_next_fire

        tz = timezone(timedelta(hours=3))
        result = compute_next_fire(time(9, 0), tz)
        dt = datetime.fromtimestamp(result, tz=tz)
        assert dt.hour == 9
        assert dt.minute == 0


# ---------------------------------------------------------------------------
# Check-in — Redis scheduling
# ---------------------------------------------------------------------------


class TestCheckinRedisOps:
    @pytest.mark.asyncio
    async def test_schedule_checkin(self) -> None:
        from companion_bot_core.proactive.checkin import schedule_checkin

        redis = AsyncMock()
        redis.zadd = AsyncMock()
        user_id = str(uuid.uuid4())
        result = await schedule_checkin(redis, user_id, time(9, 0))
        assert result > 0
        redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_unschedule_checkin(self) -> None:
        from companion_bot_core.proactive.checkin import unschedule_checkin

        redis = AsyncMock()
        redis.zrem = AsyncMock()
        user_id = str(uuid.uuid4())
        await unschedule_checkin(redis, user_id)
        redis.zrem.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_due_checkins_empty(self) -> None:
        from companion_bot_core.proactive.checkin import get_due_checkins

        redis = AsyncMock()
        redis.zrangebyscore = AsyncMock(return_value=[])
        result = await get_due_checkins(redis)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_due_checkins_with_members(self) -> None:
        from companion_bot_core.proactive.checkin import get_due_checkins

        redis = AsyncMock()
        uid1 = str(uuid.uuid4())
        uid2 = str(uuid.uuid4())
        redis.zrangebyscore = AsyncMock(return_value=[uid1.encode(), uid2.encode()])
        result = await get_due_checkins(redis)
        assert result == [uid1, uid2]

    @pytest.mark.asyncio
    async def test_mark_sent(self) -> None:
        from companion_bot_core.proactive.checkin import mark_sent

        redis = AsyncMock()
        redis.set = AsyncMock()
        user_id = str(uuid.uuid4())
        await mark_sent(redis, user_id)
        redis.set.assert_called_once()


# ---------------------------------------------------------------------------
# Scheduler — timezone parsing
# ---------------------------------------------------------------------------


class TestParseTimezone:
    def test_none_returns_utc(self) -> None:
        from companion_bot_core.proactive.scheduler import parse_timezone

        assert parse_timezone(None) == UTC

    def test_empty_returns_utc(self) -> None:
        from companion_bot_core.proactive.scheduler import parse_timezone

        assert parse_timezone("") == UTC

    def test_utc_string(self) -> None:
        from companion_bot_core.proactive.scheduler import parse_timezone

        assert parse_timezone("UTC") == UTC

    def test_gmt_string(self) -> None:
        from companion_bot_core.proactive.scheduler import parse_timezone

        assert parse_timezone("GMT") == UTC

    def test_utc_plus_offset(self) -> None:
        from companion_bot_core.proactive.scheduler import parse_timezone

        tz = parse_timezone("UTC+3")
        assert tz == timezone(timedelta(hours=3))

    def test_utc_minus_offset(self) -> None:
        from companion_bot_core.proactive.scheduler import parse_timezone

        tz = parse_timezone("UTC-5")
        assert tz == timezone(timedelta(hours=-5))

    def test_gmt_plus_offset(self) -> None:
        from companion_bot_core.proactive.scheduler import parse_timezone

        tz = parse_timezone("GMT+2")
        assert tz == timezone(timedelta(hours=2))

    def test_invalid_returns_utc(self) -> None:
        from companion_bot_core.proactive.scheduler import parse_timezone

        assert parse_timezone("America/New_York") == UTC

    def test_utc_invalid_offset_returns_utc(self) -> None:
        from companion_bot_core.proactive.scheduler import parse_timezone

        assert parse_timezone("UTC+abc") == UTC


# ---------------------------------------------------------------------------
# /checkin command — handler logic
# ---------------------------------------------------------------------------


class TestCmdCheckin:
    def _make_message(self) -> MagicMock:
        msg = MagicMock()
        msg.answer = AsyncMock()
        msg.chat = MagicMock()
        msg.chat.type = "private"
        return msg

    def _make_user(self, locale: str = "en") -> MagicMock:
        user = MagicMock()
        user.id = uuid.uuid4()
        user.locale = locale
        user.timezone = "UTC"
        return user

    def _make_profile(
        self,
        *,
        proactive_enabled: bool = False,
        checkin_time: time | None = None,
    ) -> MagicMock:
        profile = MagicMock()
        profile.proactive_enabled = proactive_enabled
        profile.checkin_time = checkin_time
        return profile

    @pytest.mark.asyncio
    async def test_status_off(self) -> None:
        from companion_bot_core.bot.handlers import cmd_checkin

        msg = self._make_message()
        user = self._make_user()
        session = AsyncMock()
        redis = AsyncMock()
        profile = self._make_profile()

        with patch(
            "companion_bot_core.bot.handlers.get_or_create_profile",
            return_value=profile,
        ):
            cmd = MagicMock()
            cmd.args = ""
            await cmd_checkin(msg, user, session, redis, cmd)

        msg.answer.assert_called_once()
        text = msg.answer.call_args[0][0]
        assert "off" in text.lower() or "выключен" in text.lower()

    @pytest.mark.asyncio
    async def test_status_on(self) -> None:
        from companion_bot_core.bot.handlers import cmd_checkin

        msg = self._make_message()
        user = self._make_user()
        session = AsyncMock()
        redis = AsyncMock()
        profile = self._make_profile(proactive_enabled=True, checkin_time=time(9, 0))

        with patch(
            "companion_bot_core.bot.handlers.get_or_create_profile",
            return_value=profile,
        ):
            cmd = MagicMock()
            cmd.args = ""
            await cmd_checkin(msg, user, session, redis, cmd)

        msg.answer.assert_called_once()
        text = msg.answer.call_args[0][0]
        assert "09:00" in text

    @pytest.mark.asyncio
    async def test_checkin_on(self) -> None:
        from companion_bot_core.bot.handlers import cmd_checkin

        msg = self._make_message()
        user = self._make_user()
        session = AsyncMock()
        session.info = {}
        redis = AsyncMock()
        redis.zadd = AsyncMock()
        profile = self._make_profile()

        with patch(
            "companion_bot_core.bot.handlers.get_or_create_profile",
            return_value=profile,
        ):
            cmd = MagicMock()
            cmd.args = "on 09:00"
            await cmd_checkin(msg, user, session, redis, cmd)

        assert profile.proactive_enabled is True
        assert profile.checkin_time == time(9, 0)
        msg.answer.assert_called_once()
        text = msg.answer.call_args[0][0]
        assert "09:00" in text

    @pytest.mark.asyncio
    async def test_checkin_off(self) -> None:
        from companion_bot_core.bot.handlers import cmd_checkin

        msg = self._make_message()
        user = self._make_user()
        session = AsyncMock()
        session.info = {}
        redis = AsyncMock()
        redis.zrem = AsyncMock()
        profile = self._make_profile(proactive_enabled=True, checkin_time=time(9, 0))

        with patch(
            "companion_bot_core.bot.handlers.get_or_create_profile",
            return_value=profile,
        ):
            cmd = MagicMock()
            cmd.args = "off"
            await cmd_checkin(msg, user, session, redis, cmd)

        assert profile.proactive_enabled is False
        assert profile.checkin_time is None

    @pytest.mark.asyncio
    async def test_checkin_on_invalid_time(self) -> None:
        from companion_bot_core.bot.handlers import cmd_checkin

        msg = self._make_message()
        user = self._make_user()
        session = AsyncMock()
        redis = AsyncMock()
        profile = self._make_profile()

        with patch(
            "companion_bot_core.bot.handlers.get_or_create_profile",
            return_value=profile,
        ):
            cmd = MagicMock()
            cmd.args = "on abcde"
            await cmd_checkin(msg, user, session, redis, cmd)

        msg.answer.assert_called_once()
        text = msg.answer.call_args[0][0]
        assert "HH:MM" in text

    @pytest.mark.asyncio
    async def test_checkin_quiet_set(self) -> None:
        from companion_bot_core.bot.handlers import cmd_checkin

        msg = self._make_message()
        user = self._make_user()
        session = AsyncMock()
        session.info = {}
        redis = AsyncMock()
        profile = self._make_profile()

        with patch(
            "companion_bot_core.bot.handlers.get_or_create_profile",
            return_value=profile,
        ):
            cmd = MagicMock()
            cmd.args = "quiet 22:00-08:00"
            await cmd_checkin(msg, user, session, redis, cmd)

        assert profile.quiet_hours_start == time(22, 0)
        assert profile.quiet_hours_end == time(8, 0)

    @pytest.mark.asyncio
    async def test_checkin_quiet_off(self) -> None:
        from companion_bot_core.bot.handlers import cmd_checkin

        msg = self._make_message()
        user = self._make_user()
        session = AsyncMock()
        session.info = {}
        redis = AsyncMock()
        profile = self._make_profile()
        profile.quiet_hours_start = time(22, 0)
        profile.quiet_hours_end = time(8, 0)

        with patch(
            "companion_bot_core.bot.handlers.get_or_create_profile",
            return_value=profile,
        ):
            cmd = MagicMock()
            cmd.args = "quiet off"
            await cmd_checkin(msg, user, session, redis, cmd)

        assert profile.quiet_hours_start is None
        assert profile.quiet_hours_end is None

    @pytest.mark.asyncio
    async def test_checkin_quiet_invalid(self) -> None:
        from companion_bot_core.bot.handlers import cmd_checkin

        msg = self._make_message()
        user = self._make_user()
        session = AsyncMock()
        redis = AsyncMock()
        profile = self._make_profile()

        with patch(
            "companion_bot_core.bot.handlers.get_or_create_profile",
            return_value=profile,
        ):
            cmd = MagicMock()
            cmd.args = "quiet invalid"
            await cmd_checkin(msg, user, session, redis, cmd)

        msg.answer.assert_called_once()
        text = msg.answer.call_args[0][0]
        assert "HH:MM" in text

    @pytest.mark.asyncio
    async def test_checkin_help_on_unknown_subcommand(self) -> None:
        from companion_bot_core.bot.handlers import cmd_checkin

        msg = self._make_message()
        user = self._make_user()
        session = AsyncMock()
        redis = AsyncMock()
        profile = self._make_profile()

        with patch(
            "companion_bot_core.bot.handlers.get_or_create_profile",
            return_value=profile,
        ):
            cmd = MagicMock()
            cmd.args = "unknown"
            await cmd_checkin(msg, user, session, redis, cmd)

        msg.answer.assert_called_once()
        text = msg.answer.call_args[0][0]
        assert "/checkin" in text


# ---------------------------------------------------------------------------
# _parse_checkin_time
# ---------------------------------------------------------------------------


class TestParseCheckinTime:
    def test_valid_time(self) -> None:
        from companion_bot_core.bot.handlers import _parse_checkin_time

        assert _parse_checkin_time("09:00") == time(9, 0)

    def test_valid_time_with_spaces(self) -> None:
        from companion_bot_core.bot.handlers import _parse_checkin_time

        assert _parse_checkin_time("  14:30  ") == time(14, 30)

    def test_empty_string(self) -> None:
        from companion_bot_core.bot.handlers import _parse_checkin_time

        assert _parse_checkin_time("") is None

    def test_no_colon(self) -> None:
        from companion_bot_core.bot.handlers import _parse_checkin_time

        assert _parse_checkin_time("0900") is None

    def test_invalid_hour(self) -> None:
        from companion_bot_core.bot.handlers import _parse_checkin_time

        assert _parse_checkin_time("25:00") is None

    def test_invalid_minute(self) -> None:
        from companion_bot_core.bot.handlers import _parse_checkin_time

        assert _parse_checkin_time("09:61") is None

    def test_non_numeric(self) -> None:
        from companion_bot_core.bot.handlers import _parse_checkin_time

        assert _parse_checkin_time("ab:cd") is None

    def test_midnight(self) -> None:
        from companion_bot_core.bot.handlers import _parse_checkin_time

        assert _parse_checkin_time("00:00") == time(0, 0)

    def test_end_of_day(self) -> None:
        from companion_bot_core.bot.handlers import _parse_checkin_time

        assert _parse_checkin_time("23:59") == time(23, 59)


# ---------------------------------------------------------------------------
# reconcile_schedule
# ---------------------------------------------------------------------------


class TestReconcileSchedule:
    @pytest.mark.asyncio
    async def test_adds_missing_user_to_schedule(self) -> None:
        from companion_bot_core.proactive.scheduler import reconcile_schedule

        redis = AsyncMock()
        redis.zscore = AsyncMock(return_value=None)
        redis.zadd = AsyncMock()

        user_id = uuid.uuid4()
        checkin_t = time(9, 0)

        engine = AsyncMock()

        mock_result = MagicMock()
        mock_result.all.return_value = [(user_id, checkin_t, "UTC+3")]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "companion_bot_core.proactive.scheduler.get_async_session",
        ) as mock_get_session:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_session.return_value = ctx

            await reconcile_schedule(redis, engine)

        redis.zscore.assert_called_once_with("checkin:schedule", str(user_id))
        redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_user_already_in_schedule_with_correct_time(self) -> None:
        from companion_bot_core.proactive.scheduler import reconcile_schedule

        redis = AsyncMock()
        redis.zadd = AsyncMock()

        user_id = uuid.uuid4()
        engine = AsyncMock()

        # Return a zscore that matches the expected fire time (within tolerance).
        expected_fire = 1772874000.0
        redis.zscore = AsyncMock(return_value=expected_fire)

        mock_result = MagicMock()
        mock_result.all.return_value = [(user_id, time(9, 0), "UTC")]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "companion_bot_core.proactive.scheduler.get_async_session",
            ) as mock_get_session,
            patch(
                "companion_bot_core.proactive.scheduler.compute_next_fire",
                return_value=expected_fire,
            ),
        ):
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_session.return_value = ctx

            await reconcile_schedule(redis, engine)

        redis.zscore.assert_called_once()
        redis.zadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_corrects_stale_fire_time(self) -> None:
        from companion_bot_core.proactive.scheduler import reconcile_schedule

        redis = AsyncMock()
        redis.zadd = AsyncMock()

        user_id = uuid.uuid4()
        engine = AsyncMock()

        # Stale score differs from expected by more than 120 seconds.
        stale_fire = 1772870000.0
        expected_fire = 1772874000.0
        redis.zscore = AsyncMock(return_value=stale_fire)

        mock_result = MagicMock()
        mock_result.all.return_value = [(user_id, time(9, 0), "UTC")]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "companion_bot_core.proactive.scheduler.get_async_session",
            ) as mock_get_session,
            patch(
                "companion_bot_core.proactive.scheduler.compute_next_fire",
                return_value=expected_fire,
            ),
        ):
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_session.return_value = ctx

            await reconcile_schedule(redis, engine)

        redis.zscore.assert_called_once()
        redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_db_error_gracefully(self) -> None:
        from companion_bot_core.proactive.scheduler import reconcile_schedule

        redis = AsyncMock()
        engine = AsyncMock()

        with patch(
            "companion_bot_core.proactive.scheduler.get_async_session",
            side_effect=RuntimeError("db down"),
        ):
            # Should not raise
            await reconcile_schedule(redis, engine)
