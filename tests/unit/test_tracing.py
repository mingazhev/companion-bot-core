"""Unit tests for the lightweight tracing module (companion_bot_core.tracing).

Tests verify that:
- Async and synchronous span context managers complete normally.
- Exceptions propagate correctly through spans.
- Span IDs are set and restored correctly when spans nest.
- get_span_id() returns the current active span ID.
"""

from __future__ import annotations

import pytest

from companion_bot_core.tracing import get_span_id, span, sync_span


class TestAsyncSpan:
    async def test_span_completes_without_error(self) -> None:
        async with span("test.operation"):
            pass

    async def test_span_with_extra_kwargs(self) -> None:
        async with span("test.operation", user_id="abc123", model="gpt-4o"):
            pass

    async def test_span_propagates_exception(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            async with span("test.failing_operation"):
                raise ValueError("boom")

    async def test_span_id_set_inside_span(self) -> None:
        assert get_span_id() == ""
        inside_sid: str = ""
        async with span("test.check_span_id"):
            inside_sid = get_span_id()
        assert inside_sid != ""
        assert len(inside_sid) == 16  # hex chars

    async def test_span_id_restored_after_span(self) -> None:
        assert get_span_id() == ""
        async with span("test.restore"):
            pass
        assert get_span_id() == ""

    async def test_nested_spans_have_different_ids(self) -> None:
        outer_id: str = ""
        inner_id: str = ""

        async with span("outer"):
            outer_id = get_span_id()
            async with span("inner"):
                inner_id = get_span_id()
            # After inner span, outer ID is restored
            assert get_span_id() == outer_id

        assert outer_id != ""
        assert inner_id != ""
        assert outer_id != inner_id

    async def test_span_id_restored_after_exception(self) -> None:
        assert get_span_id() == ""
        with pytest.raises(RuntimeError):
            async with span("test.error_span"):
                raise RuntimeError("intentional")
        assert get_span_id() == ""


class TestSyncSpan:
    def test_sync_span_completes_without_error(self) -> None:
        with sync_span("test.sync_operation"):
            pass

    def test_sync_span_with_extra_kwargs(self) -> None:
        with sync_span("test.sync_operation", key="value"):
            pass

    def test_sync_span_propagates_exception(self) -> None:
        with pytest.raises(ValueError, match="sync_error"), sync_span("test.sync_failing"):
            raise ValueError("sync_error")

    def test_sync_span_id_set_inside_span(self) -> None:
        inside_sid: str = ""
        with sync_span("test.sync_check"):
            inside_sid = get_span_id()
        assert inside_sid != ""
        assert len(inside_sid) == 16

    def test_sync_span_id_restored_after_span(self) -> None:
        assert get_span_id() == ""
        with sync_span("test.sync_restore"):
            pass
        assert get_span_id() == ""

    def test_sync_nested_spans(self) -> None:
        outer_id: str = ""
        inner_id: str = ""

        with sync_span("sync_outer"):
            outer_id = get_span_id()
            with sync_span("sync_inner"):
                inner_id = get_span_id()
            assert get_span_id() == outer_id

        assert outer_id != inner_id

    def test_sync_span_id_restored_after_exception(self) -> None:
        with pytest.raises(KeyError), sync_span("test.sync_error"):
            raise KeyError("gone")
        assert get_span_id() == ""


class TestGetSpanId:
    def test_returns_empty_outside_span(self) -> None:
        assert get_span_id() == ""

    async def test_returns_nonempty_inside_async_span(self) -> None:
        async with span("test.get_span_id"):
            assert get_span_id() != ""

    def test_returns_nonempty_inside_sync_span(self) -> None:
        with sync_span("test.get_span_id_sync"):
            assert get_span_id() != ""
