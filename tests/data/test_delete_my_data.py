"""Data integrity tests: /delete_my_data removes personal records and preserves audit.

These tests verify the hard-delete contract:
- An audit log entry is written BEFORE the user row is deleted (ordering guarantee).
- The audit entry carries the correct event_type ("user_data_deleted") and reason.
- The DELETE statement targets the users table; cascade rules in the schema then
  remove all linked personal data (profiles, snapshots, messages, events, jobs).
- The audit_log table uses ON DELETE SET NULL, so the audit trail is preserved
  after the user row is removed.
- The function is idempotent (no exception if the user no longer exists).
- The function never commits — committing is always the caller's responsibility.

Tests use a mock AsyncSession.  SQL statement analysis verifies table targeting
without a live database, following the project's established unit-test patterns.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from tdbot.db.models import AuditLog
from tdbot.privacy.delete_user import hard_delete_user

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock())
    return session


# ---------------------------------------------------------------------------
# Audit log entry is created with correct fields
# ---------------------------------------------------------------------------


class TestAuditLogEntry:
    """The function must write an audit log entry before deleting the user row."""

    async def test_audit_entry_is_added(self) -> None:
        session = _make_session()
        user_id = uuid.uuid4()
        await hard_delete_user(user_id, session)

        session.add.assert_called_once()
        added = session.add.call_args[0][0]
        assert isinstance(added, AuditLog)

    async def test_audit_entry_event_type_is_user_data_deleted(self) -> None:
        session = _make_session()
        user_id = uuid.uuid4()
        await hard_delete_user(user_id, session)

        added = session.add.call_args[0][0]
        assert added.event_type == "user_data_deleted"

    async def test_audit_entry_user_id_matches(self) -> None:
        """The audit entry must record the user_id before the row is removed."""
        session = _make_session()
        user_id = uuid.uuid4()
        await hard_delete_user(user_id, session)

        added = session.add.call_args[0][0]
        assert added.user_id == user_id

    async def test_audit_entry_reason_is_user_request(self) -> None:
        session = _make_session()
        user_id = uuid.uuid4()
        await hard_delete_user(user_id, session)

        added = session.add.call_args[0][0]
        assert isinstance(added.details_json, dict)
        assert added.details_json.get("reason") == "user_request"

    async def test_audit_entry_details_json_is_dict(self) -> None:
        session = _make_session()
        await hard_delete_user(uuid.uuid4(), session)

        added = session.add.call_args[0][0]
        assert isinstance(added.details_json, dict)


# ---------------------------------------------------------------------------
# Audit entry is written BEFORE the DELETE (ordering guarantee)
# ---------------------------------------------------------------------------


class TestAuditWrittenBeforeDelete:
    """Session.add() must be called before session.execute() to guarantee the
    audit trail is preserved even if the DELETE fails partway through."""

    async def test_add_precedes_execute(self) -> None:
        call_order: list[str] = []

        session = AsyncMock()

        def track_add(obj: Any) -> None:
            call_order.append("add")

        async def track_execute(stmt: Any) -> MagicMock:
            call_order.append("execute")
            return MagicMock()

        session.add = MagicMock(side_effect=track_add)
        session.execute.side_effect = track_execute

        await hard_delete_user(uuid.uuid4(), session)

        assert call_order == ["add", "execute"], (
            "Audit log entry must be added before the DELETE is executed"
        )


# ---------------------------------------------------------------------------
# DELETE targets the users table (cascade handles linked personal data)
# ---------------------------------------------------------------------------


class TestDeleteTargetsUsersTable:
    """The DELETE statement must target the users table so ON DELETE CASCADE
    automatically removes all linked personal data tables."""

    async def test_delete_statement_targets_users(self) -> None:
        captured: list[Any] = []
        session = AsyncMock()
        session.add = MagicMock()

        async def capture(stmt: Any) -> MagicMock:
            captured.append(stmt)
            return MagicMock()

        session.execute.side_effect = capture
        await hard_delete_user(uuid.uuid4(), session)

        assert len(captured) == 1
        sql = str(captured[0].compile(compile_kwargs={"literal_binds": False})).lower()
        assert "users" in sql

    async def test_delete_statement_is_a_delete(self) -> None:
        captured: list[Any] = []
        session = AsyncMock()
        session.add = MagicMock()

        async def capture(stmt: Any) -> MagicMock:
            captured.append(stmt)
            return MagicMock()

        session.execute.side_effect = capture
        await hard_delete_user(uuid.uuid4(), session)

        sql = str(captured[0].compile(compile_kwargs={"literal_binds": False})).lower()
        assert sql.lstrip().startswith("delete")

    async def test_delete_is_scoped_to_specific_user(self) -> None:
        """DELETE must be parameterised by user_id, not a blanket delete."""
        captured: list[Any] = []
        session = AsyncMock()
        session.add = MagicMock()

        async def capture(stmt: Any) -> MagicMock:
            captured.append(stmt)
            return MagicMock()

        session.execute.side_effect = capture
        user_id = uuid.uuid4()
        await hard_delete_user(user_id, session)

        sql = str(captured[0].compile(compile_kwargs={"literal_binds": False})).lower()
        # WHERE clause must be present
        assert "where" in sql


# ---------------------------------------------------------------------------
# Audit minimality: AuditLog schema allows user_id to be NULL
# ---------------------------------------------------------------------------


class TestAuditMinimality:
    """The AuditLog model must allow user_id=None so the row survives after
    the user is hard-deleted (ON DELETE SET NULL database constraint)."""

    def test_audit_log_user_id_is_nullable(self) -> None:
        """Constructing an AuditLog with user_id=None must succeed."""
        entry = AuditLog(
            user_id=None,
            event_type="user_data_deleted",
            details_json={"reason": "user_request"},
        )
        assert entry.user_id is None

    def test_audit_log_preserves_event_type_without_user(self) -> None:
        entry = AuditLog(
            user_id=None,
            event_type="user_data_deleted",
            details_json={"reason": "user_request"},
        )
        assert entry.event_type == "user_data_deleted"

    def test_audit_log_can_carry_details_without_user_id(self) -> None:
        entry = AuditLog(
            user_id=None,
            event_type="user_data_deleted",
            details_json={"reason": "user_request", "retained_for": "legal_minimality"},
        )
        assert entry.details_json["retained_for"] == "legal_minimality"


# ---------------------------------------------------------------------------
# Caller responsibility: function must not commit
# ---------------------------------------------------------------------------


class TestNoAutoCommit:
    async def test_does_not_commit(self) -> None:
        session = _make_session()
        await hard_delete_user(uuid.uuid4(), session)
        session.commit.assert_not_called()

    async def test_does_not_rollback(self) -> None:
        session = _make_session()
        await hard_delete_user(uuid.uuid4(), session)
        session.rollback.assert_not_called()


# ---------------------------------------------------------------------------
# Idempotency: no exception when user row does not exist
# ---------------------------------------------------------------------------


class TestIdempotency:
    async def test_no_exception_for_nonexistent_user(self) -> None:
        """DELETE on a missing row is a no-op at the DB level; function must not raise."""
        session = _make_session()
        # Simulate DB returning rowcount=0 (user not found)
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute.return_value = mock_result

        # Must not raise
        await hard_delete_user(uuid.uuid4(), session)

    async def test_audit_log_still_written_even_if_user_missing(self) -> None:
        """Audit entry is always written regardless of whether user existed."""
        session = _make_session()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute.return_value = mock_result

        await hard_delete_user(uuid.uuid4(), session)

        # Audit entry must still have been added
        session.add.assert_called_once()
        added = session.add.call_args[0][0]
        assert isinstance(added, AuditLog)
