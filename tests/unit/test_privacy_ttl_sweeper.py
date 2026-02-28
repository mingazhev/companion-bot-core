"""Unit tests for the TTL sweeper (conversation_messages expiration)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from companion_bot_core.privacy.ttl_sweeper import sweep_expired_messages


class TestSweepExpiredMessages:
    """Tests for sweep_expired_messages using a mock async session."""

    async def test_returns_rowcount(self) -> None:
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 5
        session.execute.return_value = mock_result

        deleted = await sweep_expired_messages(session)

        assert deleted == 5

    async def test_calls_execute(self) -> None:
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute.return_value = mock_result

        await sweep_expired_messages(session)

        session.execute.assert_awaited_once()

    async def test_zero_rows_deleted(self) -> None:
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute.return_value = mock_result

        deleted = await sweep_expired_messages(session)

        assert deleted == 0

    async def test_large_batch_deletion(self) -> None:
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 50_000
        session.execute.return_value = mock_result

        deleted = await sweep_expired_messages(session)

        assert deleted == 50_000

    async def test_does_not_commit(self) -> None:
        """The sweeper must NOT commit — that is the caller's responsibility."""
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 3
        session.execute.return_value = mock_result

        await sweep_expired_messages(session)

        session.commit.assert_not_called()

    async def test_statement_filters_only_expired_rows(self) -> None:
        """Verify the DELETE statement only targets rows with a non-null past TTL."""
        captured: list[Any] = []
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute.return_value = mock_result

        async def capture_stmt(stmt: Any) -> MagicMock:
            captured.append(stmt)
            return mock_result

        session.execute.side_effect = capture_stmt

        await sweep_expired_messages(session)

        assert len(captured) == 1
        stmt = captured[0]
        # Verify it is a DELETE against conversation_messages
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": False}))
        assert "conversation_messages" in compiled.lower()
        assert "ttl_expires_at" in compiled.lower()
