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


def _make_session(user_exists: bool = True, user_id: uuid.UUID | None = None) -> AsyncMock:
    """Create a mock session where the user existence check returns *user_exists*.

    The first ``execute()`` call (the SELECT check) returns a result whose
    ``scalar_one_or_none()`` returns *user_id* (truthy) when *user_exists*,
    or ``None`` otherwise.  Subsequent calls return a plain MagicMock.
    """
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = user_id if user_exists else None
    delete_result = MagicMock()

    session = AsyncMock()
    session.add = MagicMock()
    session.execute = AsyncMock(side_effect=[select_result, delete_result])
    return session


# ---------------------------------------------------------------------------
# Audit log entry is created with correct fields
# ---------------------------------------------------------------------------


class TestAuditLogEntry:
    """The function must write an audit log entry before deleting the user row."""

    async def test_audit_entry_is_added(self) -> None:
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)
        await hard_delete_user(user_id, session)

        session.add.assert_called_once()
        added = session.add.call_args[0][0]
        assert isinstance(added, AuditLog)

    async def test_audit_entry_event_type_is_user_data_deleted(self) -> None:
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)
        await hard_delete_user(user_id, session)

        added = session.add.call_args[0][0]
        assert added.event_type == "user_data_deleted"

    async def test_audit_entry_user_id_matches(self) -> None:
        """The audit entry must record the user_id before the row is removed."""
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)
        await hard_delete_user(user_id, session)

        added = session.add.call_args[0][0]
        assert added.user_id == user_id

    async def test_audit_entry_reason_is_user_request(self) -> None:
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)
        await hard_delete_user(user_id, session)

        added = session.add.call_args[0][0]
        assert isinstance(added.details_json, dict)
        assert added.details_json.get("reason") == "user_request"

    async def test_audit_entry_details_json_is_dict(self) -> None:
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)
        await hard_delete_user(user_id, session)

        added = session.add.call_args[0][0]
        assert isinstance(added.details_json, dict)


# ---------------------------------------------------------------------------
# Audit entry is written BEFORE the DELETE (ordering guarantee)
# ---------------------------------------------------------------------------


class TestAuditWrittenBeforeDelete:
    """Session.add() must be called before the DELETE execute() to guarantee the
    audit trail is preserved even if the DELETE fails partway through."""

    async def test_add_precedes_execute(self) -> None:
        call_order: list[str] = []
        user_id = uuid.uuid4()

        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = user_id

        session = AsyncMock()
        execute_call_count = 0

        def track_add(obj: Any) -> None:
            call_order.append("add")

        async def track_execute(stmt: Any) -> MagicMock:
            nonlocal execute_call_count
            execute_call_count += 1
            if execute_call_count == 1:
                # First call is the SELECT existence check
                return select_result
            call_order.append("delete")
            return MagicMock()

        session.add = MagicMock(side_effect=track_add)
        session.execute = AsyncMock(side_effect=track_execute)

        await hard_delete_user(user_id, session)

        assert call_order == ["add", "delete"], (
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
        user_id = uuid.uuid4()

        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = user_id

        session = AsyncMock()
        session.add = MagicMock()
        call_count = 0

        async def capture(stmt: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return select_result
            captured.append(stmt)
            return MagicMock()

        session.execute = AsyncMock(side_effect=capture)
        await hard_delete_user(user_id, session)

        assert len(captured) == 1
        sql = str(captured[0].compile(compile_kwargs={"literal_binds": False})).lower()
        assert "users" in sql

    async def test_delete_statement_is_a_delete(self) -> None:
        captured: list[Any] = []
        user_id = uuid.uuid4()

        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = user_id

        session = AsyncMock()
        session.add = MagicMock()
        call_count = 0

        async def capture(stmt: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return select_result
            captured.append(stmt)
            return MagicMock()

        session.execute = AsyncMock(side_effect=capture)
        await hard_delete_user(user_id, session)

        sql = str(captured[0].compile(compile_kwargs={"literal_binds": False})).lower()
        assert sql.lstrip().startswith("delete")

    async def test_delete_is_scoped_to_specific_user(self) -> None:
        """DELETE must be parameterised by user_id, not a blanket delete."""
        captured: list[Any] = []
        user_id = uuid.uuid4()

        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = user_id

        session = AsyncMock()
        session.add = MagicMock()
        call_count = 0

        async def capture(stmt: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return select_result
            captured.append(stmt)
            return MagicMock()

        session.execute = AsyncMock(side_effect=capture)
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
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)
        await hard_delete_user(user_id, session)
        session.commit.assert_not_called()

    async def test_does_not_rollback(self) -> None:
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)
        await hard_delete_user(user_id, session)
        session.rollback.assert_not_called()


# ---------------------------------------------------------------------------
# Idempotency: no exception when user row does not exist
# ---------------------------------------------------------------------------


class TestIdempotency:
    async def test_no_exception_for_nonexistent_user(self) -> None:
        """DELETE on a missing row is a no-op; function must not raise."""
        session = _make_session(user_exists=False)
        # Must not raise
        await hard_delete_user(uuid.uuid4(), session)

    async def test_no_audit_entry_when_user_missing(self) -> None:
        """When the user row is already gone, no audit entry is written
        (inserting one would violate the FK constraint on audit_log.user_id)."""
        session = _make_session(user_exists=False)
        await hard_delete_user(uuid.uuid4(), session)
        session.add.assert_not_called()

    async def test_redis_keys_still_cleaned_when_user_missing(self) -> None:
        """Even if the user row is gone, Redis keys should be cleaned up."""
        user_id = uuid.uuid4()
        session = _make_session(user_exists=False)
        redis = AsyncMock()

        await hard_delete_user(user_id, session, redis=redis, telegram_user_id=12345)

        session.add.assert_not_called()
        redis.delete.assert_awaited_once()
