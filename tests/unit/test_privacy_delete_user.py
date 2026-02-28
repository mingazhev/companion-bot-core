"""Unit tests for the hard_delete_user flow."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from companion_bot_core.privacy.delete_user import hard_delete_user


def _make_session(user_exists: bool = True, user_id: uuid.UUID | None = None) -> AsyncMock:
    """Create a mock session where the user existence check returns *user_exists*.

    The first ``execute()`` call (the SELECT existence check) returns a result
    whose ``scalar_one_or_none()`` returns the *user_id* (if *user_exists*) or
    ``None``.  Subsequent calls (the DELETE) return a plain ``MagicMock``.
    """
    # Result object for the SELECT query
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = user_id if user_exists else None

    # Build side_effect list: first call -> select result, rest -> default mock
    delete_result = MagicMock()

    session = AsyncMock()
    session.add = MagicMock()
    session.execute = AsyncMock(side_effect=[select_result, delete_result])
    return session


class TestHardDeleteUser:
    """Tests for hard_delete_user using a mock async session."""

    async def test_adds_audit_log_entry(self) -> None:
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)

        await hard_delete_user(user_id, session)

        session.add.assert_called_once()
        added = session.add.call_args[0][0]

        from companion_bot_core.db.models import AuditLog

        assert isinstance(added, AuditLog)
        assert added.user_id == user_id
        assert added.event_type == "user_data_deleted"

    async def test_audit_details_json(self) -> None:
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)

        await hard_delete_user(user_id, session)

        added = session.add.call_args[0][0]
        assert added.details_json["reason"] == "user_request"

    async def test_executes_select_and_delete(self) -> None:
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)

        await hard_delete_user(user_id, session)

        # Two execute calls: SELECT existence check + DELETE
        assert session.execute.await_count == 2

    async def test_audit_added_before_delete(self) -> None:
        """Verify session.add() is called before the DELETE execute()."""
        call_order: list[str] = []
        user_id = uuid.uuid4()

        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = user_id

        session = AsyncMock()

        def track_add(obj: Any) -> None:
            call_order.append("add")

        execute_call_count = 0

        async def track_execute(stmt: Any) -> MagicMock:
            nonlocal execute_call_count
            execute_call_count += 1
            if execute_call_count == 1:
                # First call is the SELECT check
                return select_result
            call_order.append("delete")
            return MagicMock()

        session.add = MagicMock(side_effect=track_add)
        session.execute = AsyncMock(side_effect=track_execute)

        await hard_delete_user(user_id, session)

        assert call_order == ["add", "delete"]

    async def test_does_not_commit(self) -> None:
        """Committing must be the caller's responsibility."""
        user_id = uuid.uuid4()
        session = _make_session(user_exists=True, user_id=user_id)

        await hard_delete_user(user_id, session)

        session.commit.assert_not_called()

    async def test_delete_statement_targets_users_table(self) -> None:
        """Verify the DELETE is issued against the users table."""
        captured: list[Any] = []
        user_id = uuid.uuid4()

        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = user_id

        session = AsyncMock()
        session.add = MagicMock()

        call_count = 0

        async def capture_stmt(stmt: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return select_result
            captured.append(stmt)
            return MagicMock()

        session.execute = AsyncMock(side_effect=capture_stmt)

        await hard_delete_user(user_id, session)

        assert len(captured) == 1
        compiled = str(captured[0].compile(compile_kwargs={"literal_binds": False}))
        assert "users" in compiled.lower()

    async def test_idempotent_on_missing_user(self) -> None:
        """If the user row does not exist the function should not raise."""
        session = _make_session(user_exists=False)

        # Should not raise even if user does not exist
        await hard_delete_user(uuid.uuid4(), session)

        # No audit entry or DELETE should be issued for a missing user
        session.add.assert_not_called()

    async def test_missing_user_still_cleans_redis_keys(self) -> None:
        """Even if the user row is gone, Redis keys should be cleaned up."""
        user_id = uuid.uuid4()
        session = _make_session(user_exists=False)
        redis = AsyncMock()

        await hard_delete_user(user_id, session, redis=redis, telegram_user_id=12345)

        session.add.assert_not_called()
        redis.delete.assert_awaited_once()
