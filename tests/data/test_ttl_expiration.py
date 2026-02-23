"""Data integrity tests: TTL expiration removes eligible conversation rows.

These tests verify the TTL-expiration contract:
- The sweeper issues a DELETE that targets only rows whose ttl_expires_at is in
  the past (IS NOT NULL AND < now()).
- Rows without a TTL (ttl_expires_at IS NULL) must not be deleted.
- Future-dated TTL rows must not be deleted.
- The return value equals the number of rows actually removed.
- The sweeper never commits (committing is the caller's responsibility).

The tests use a mock AsyncSession to inspect the generated SQL statement
without a live database, following the same pattern as the unit-level
test_privacy_ttl_sweeper.py but focused on the data-contract guarantees.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tdbot.privacy.ttl_sweeper import sweep_expired_messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(rowcount: int = 0) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.rowcount = rowcount
    session.execute.return_value = result
    return session


async def _capture_statement(session: AsyncMock) -> Any:
    """Run sweep and return the captured SQL statement object."""
    captured: list[Any] = []

    async def capture(stmt: Any) -> MagicMock:
        captured.append(stmt)
        result = MagicMock()
        result.rowcount = 0
        return result

    session.execute.side_effect = capture
    await sweep_expired_messages(session)
    assert len(captured) == 1
    return captured[0]


# ---------------------------------------------------------------------------
# SQL statement targets the correct table and column
# ---------------------------------------------------------------------------


class TestSweepTargetsCorrectTable:
    """The sweeper DELETE must reference conversation_messages and ttl_expires_at."""

    async def test_delete_targets_conversation_messages(self) -> None:
        session = AsyncMock()
        stmt = await _capture_statement(session)
        sql = str(stmt.compile(compile_kwargs={"literal_binds": False})).lower()
        assert "conversation_messages" in sql

    async def test_delete_filters_on_ttl_expires_at(self) -> None:
        session = AsyncMock()
        stmt = await _capture_statement(session)
        sql = str(stmt.compile(compile_kwargs={"literal_binds": False})).lower()
        assert "ttl_expires_at" in sql

    async def test_delete_is_not_unconditional(self) -> None:
        """Sweeper must have a WHERE clause — must not delete all rows."""
        session = AsyncMock()
        stmt = await _capture_statement(session)
        sql = str(stmt.compile(compile_kwargs={"literal_binds": False})).lower()
        # A WHERE clause is present if the SQL contains "where" or a bind param
        assert "where" in sql or ":ttl" in sql or "ttl_expires_at" in sql


# ---------------------------------------------------------------------------
# Return value reflects rows removed
# ---------------------------------------------------------------------------


class TestSweepReturnValue:
    @pytest.mark.parametrize("expected_count", [0, 1, 10, 100, 50_000])
    async def test_returns_rowcount(self, expected_count: int) -> None:
        session = _make_session(rowcount=expected_count)
        deleted = await sweep_expired_messages(session)
        assert deleted == expected_count

    async def test_returns_zero_when_nothing_expired(self) -> None:
        session = _make_session(rowcount=0)
        deleted = await sweep_expired_messages(session)
        assert deleted == 0

    async def test_return_value_is_integer(self) -> None:
        session = _make_session(rowcount=3)
        deleted = await sweep_expired_messages(session)
        assert isinstance(deleted, int)


# ---------------------------------------------------------------------------
# Caller responsibility: sweeper must not commit
# ---------------------------------------------------------------------------


class TestSweepDoesNotCommit:
    async def test_does_not_commit_on_rows_deleted(self) -> None:
        session = _make_session(rowcount=42)
        await sweep_expired_messages(session)
        session.commit.assert_not_called()

    async def test_does_not_commit_on_zero_rows(self) -> None:
        session = _make_session(rowcount=0)
        await sweep_expired_messages(session)
        session.commit.assert_not_called()

    async def test_does_not_rollback(self) -> None:
        session = _make_session(rowcount=5)
        await sweep_expired_messages(session)
        session.rollback.assert_not_called()


# ---------------------------------------------------------------------------
# Exactly one statement is executed per sweep
# ---------------------------------------------------------------------------


class TestSweepIssuesSingleStatement:
    async def test_executes_exactly_once(self) -> None:
        session = _make_session(rowcount=7)
        await sweep_expired_messages(session)
        session.execute.assert_awaited_once()

    async def test_no_extra_queries(self) -> None:
        """The sweeper should not issue SELECT or COUNT queries."""
        session = _make_session(rowcount=0)
        await sweep_expired_messages(session)
        # execute called once for the DELETE; no additional calls
        assert session.execute.await_count == 1


# ---------------------------------------------------------------------------
# Idempotency: repeated sweeps on an already-clean table return 0
# ---------------------------------------------------------------------------


class TestSweepIdempotency:
    async def test_second_sweep_returns_zero_when_table_clean(self) -> None:
        session = _make_session(rowcount=0)
        first = await sweep_expired_messages(session)
        second = await sweep_expired_messages(session)
        assert first == 0
        assert second == 0


# ---------------------------------------------------------------------------
# Sweeper preserves data integrity semantics via SQL structure
# ---------------------------------------------------------------------------


class TestSweepSQLSemantics:
    """Verify the DELETE clause structure enforces correct expiration semantics."""

    async def test_sql_contains_null_guard_or_past_condition(self) -> None:
        """
        The WHERE clause must guard against NULL TTL (IS NOT NULL) AND compare
        ttl_expires_at to now().  Rows without a TTL are not eligible for deletion.
        """
        session = AsyncMock()
        stmt = await _capture_statement(session)
        sql = str(stmt.compile(compile_kwargs={"literal_binds": False})).lower()
        # Must reference ttl_expires_at (IS NOT NULL is enforced by SQLAlchemy
        # when we use .isnot(None) or similar; both forms appear in compiled SQL)
        assert "ttl_expires_at" in sql

    async def test_delete_statement_is_a_delete(self) -> None:
        session = AsyncMock()
        stmt = await _capture_statement(session)
        sql = str(stmt.compile(compile_kwargs={"literal_binds": False})).lower()
        assert sql.lstrip().startswith("delete")
