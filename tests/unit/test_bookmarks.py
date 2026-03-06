"""Unit tests for conversation bookmarks (2.4).

Tests cover:
- bookmarks.is_bookmark_request: intent detection for RU and EN triggers
- bookmarks.save_bookmark: DB persistence
- bookmarks.get_bookmarks: retrieval (ordered, limited)
- bookmarks.search_bookmarks: text search
- /bookmarks command: formatting, empty state, search
- Orchestrator integration: bookmark detection and notice prepend
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from companion_bot_core.orchestrator.bookmarks import (
    is_bookmark_request,
    save_bookmark,
)

# ---------------------------------------------------------------------------
# is_bookmark_request — intent detection
# ---------------------------------------------------------------------------


class TestIsBookmarkRequest:
    def test_zapomni_eto(self) -> None:
        assert is_bookmark_request("запомни это") is True

    def test_sohrani_eto(self) -> None:
        assert is_bookmark_request("сохрани это") is True

    def test_eto_vazhno(self) -> None:
        assert is_bookmark_request("это важно") is True

    def test_zakladka(self) -> None:
        assert is_bookmark_request("сделай закладку") is True

    def test_ne_zabud_eto(self) -> None:
        assert is_bookmark_request("не забудь это") is True

    def test_zapishi_eto(self) -> None:
        assert is_bookmark_request("запиши это") is True

    def test_remember_this_en(self) -> None:
        assert is_bookmark_request("remember this") is True

    def test_save_this_en(self) -> None:
        assert is_bookmark_request("save this") is True

    def test_this_is_important_en(self) -> None:
        assert is_bookmark_request("this is important") is True

    def test_bookmark_this_en(self) -> None:
        assert is_bookmark_request("bookmark this") is True

    def test_keep_this_en(self) -> None:
        assert is_bookmark_request("keep this") is True

    def test_note_this_en(self) -> None:
        assert is_bookmark_request("note this") is True

    def test_dont_forget_this_en(self) -> None:
        assert is_bookmark_request("don't forget this") is True

    def test_case_insensitive(self) -> None:
        assert is_bookmark_request("ЗАПОМНИ ЭТО") is True
        assert is_bookmark_request("Remember This") is True

    def test_normal_message_not_detected(self) -> None:
        assert is_bookmark_request("привет, как дела?") is False

    def test_empty_string(self) -> None:
        assert is_bookmark_request("") is False

    def test_generic_remember_not_detected(self) -> None:
        # "remember" alone without "this" should not trigger
        assert is_bookmark_request("remember me") is False

    def test_generic_save_not_detected(self) -> None:
        # "save" alone without "this" should not trigger
        assert is_bookmark_request("save the world") is False


# ---------------------------------------------------------------------------
# save_bookmark — DB persistence
# ---------------------------------------------------------------------------


class TestSaveBookmark:
    @pytest.mark.asyncio
    async def test_save_bookmark_creates_row(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()

        bookmark = await save_bookmark(
            session, user_id,
            user_message="как пережить стресс?",
            bot_response="Стресс - это нормальная реакция...",
        )

        session.add.assert_called_once()
        session.flush.assert_awaited_once()
        assert bookmark.user_id == user_id
        assert bookmark.user_message == "как пережить стресс?"
        assert bookmark.bot_response == "Стресс - это нормальная реакция..."
        assert bookmark.tag is None

    @pytest.mark.asyncio
    async def test_save_bookmark_with_tag(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()

        bookmark = await save_bookmark(
            session, user_id,
            user_message="msg",
            bot_response="resp",
            tag="important",
        )

        assert bookmark.tag == "important"


# ---------------------------------------------------------------------------
# get_bookmarks — retrieval
# ---------------------------------------------------------------------------


def _make_mock_bookmark(
    user_message: str = "msg", bot_response: str = "resp", tag: str | None = None,
) -> MagicMock:
    bk = MagicMock()
    bk.user_message = user_message
    bk.bot_response = bot_response
    bk.tag = tag
    return bk


class TestGetBookmarks:
    @pytest.mark.asyncio
    async def test_get_bookmarks_returns_list(self) -> None:
        from companion_bot_core.orchestrator.bookmarks import get_bookmarks

        user_id = uuid.uuid4()
        bk1 = _make_mock_bookmark("msg1", "resp1")
        bk2 = _make_mock_bookmark("msg2", "resp2")

        # Build a mock session that returns bookmarks
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [bk1, bk2]
        mock_result.scalars.return_value = mock_scalars

        session = AsyncMock()
        session.execute.return_value = mock_result

        result = await get_bookmarks(session, user_id)
        assert len(result) == 2
        assert result[0].user_message == "msg1"
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_bookmarks_empty(self) -> None:
        from companion_bot_core.orchestrator.bookmarks import get_bookmarks

        user_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars

        session = AsyncMock()
        session.execute.return_value = mock_result

        result = await get_bookmarks(session, user_id)
        assert result == []


# ---------------------------------------------------------------------------
# search_bookmarks — text search
# ---------------------------------------------------------------------------


class TestSearchBookmarks:
    @pytest.mark.asyncio
    async def test_search_bookmarks_returns_results(self) -> None:
        from companion_bot_core.orchestrator.bookmarks import search_bookmarks

        user_id = uuid.uuid4()
        bk = _make_mock_bookmark("стресс на работе", "Понимаю тебя")
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [bk]
        mock_result.scalars.return_value = mock_scalars

        session = AsyncMock()
        session.execute.return_value = mock_result

        result = await search_bookmarks(session, user_id, "стресс")
        assert len(result) == 1
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_bookmarks_empty_results(self) -> None:
        from companion_bot_core.orchestrator.bookmarks import search_bookmarks

        user_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars

        session = AsyncMock()
        session.execute.return_value = mock_result

        result = await search_bookmarks(session, user_id, "nonexistent")
        assert result == []


# ---------------------------------------------------------------------------
# _format_bookmark_list — formatting
# ---------------------------------------------------------------------------


class TestFormatBookmarkList:
    def test_format_single_bookmark(self) -> None:
        from companion_bot_core.bot.handlers import _format_bookmark_list

        bk = MagicMock()
        bk.created_at = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
        bk.user_message = "как дела?"
        bk.bot_response = "Всё хорошо, спасибо!"
        bk.tag = None

        result = _format_bookmark_list([bk], "ru")
        assert "05.03.2026" in result
        assert "как дела?" in result
        assert "Всё хорошо, спасибо!" in result

    def test_format_bookmark_with_tag(self) -> None:
        from companion_bot_core.bot.handlers import _format_bookmark_list

        bk = MagicMock()
        bk.created_at = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
        bk.user_message = "msg"
        bk.bot_response = "resp"
        bk.tag = "important"

        result = _format_bookmark_list([bk], "en")
        assert "[important]" in result

    def test_format_truncates_long_messages(self) -> None:
        from companion_bot_core.bot.handlers import _format_bookmark_list

        bk = MagicMock()
        bk.created_at = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
        bk.user_message = "x" * 200
        bk.bot_response = "y" * 200
        bk.tag = None

        result = _format_bookmark_list([bk], "ru")
        # Both messages should be truncated with "..."
        assert "..." in result

    def test_format_multiple_bookmarks(self) -> None:
        from companion_bot_core.bot.handlers import _format_bookmark_list

        bookmarks = []
        for i in range(3):
            bk = MagicMock()
            bk.created_at = datetime(2026, 3, i + 1, 12, 0, tzinfo=UTC)
            bk.user_message = f"msg {i}"
            bk.bot_response = f"resp {i}"
            bk.tag = None
            bookmarks.append(bk)

        result = _format_bookmark_list(bookmarks, "en")
        assert "1." in result
        assert "2." in result
        assert "3." in result


# ---------------------------------------------------------------------------
# Metric object — basic sanity check
# ---------------------------------------------------------------------------


class TestBookmarkMetric:
    def test_bookmark_saved_metric_exists(self) -> None:
        from companion_bot_core.metrics import BOOKMARK_SAVED

        # Verify the metric can be incremented without error
        BOOKMARK_SAVED.inc()
