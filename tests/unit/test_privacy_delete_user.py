"""Unit tests for the hard_delete_user flow."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from tdbot.privacy.delete_user import hard_delete_user


class TestHardDeleteUser:
    """Tests for hard_delete_user using a mock async session."""

    async def test_adds_audit_log_entry(self) -> None:
        session = AsyncMock()
        session.add = MagicMock()
        session.execute = AsyncMock()

        user_id = uuid.uuid4()
        await hard_delete_user(user_id, session)

        session.add.assert_called_once()
        added = session.add.call_args[0][0]

        from tdbot.db.models import AuditLog

        assert isinstance(added, AuditLog)
        assert added.user_id == user_id
        assert added.event_type == "user_data_deleted"

    async def test_audit_details_json(self) -> None:
        session = AsyncMock()
        session.add = MagicMock()
        session.execute = AsyncMock()

        user_id = uuid.uuid4()
        await hard_delete_user(user_id, session)

        added = session.add.call_args[0][0]
        assert added.details_json["reason"] == "user_request"

    async def test_executes_delete_statement(self) -> None:
        session = AsyncMock()
        session.add = MagicMock()
        session.execute = AsyncMock()

        user_id = uuid.uuid4()
        await hard_delete_user(user_id, session)

        session.execute.assert_awaited_once()

    async def test_audit_added_before_delete(self) -> None:
        """Verify session.add() is called before session.execute()."""
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

        assert call_order == ["add", "execute"]

    async def test_does_not_commit(self) -> None:
        """Committing must be the caller's responsibility."""
        session = AsyncMock()
        session.add = MagicMock()
        session.execute = AsyncMock()

        await hard_delete_user(uuid.uuid4(), session)

        session.commit.assert_not_called()

    async def test_delete_statement_targets_users_table(self) -> None:
        """Verify the DELETE is issued against the users table."""
        captured: list[Any] = []
        session = AsyncMock()
        session.add = MagicMock()

        async def capture_stmt(stmt: Any) -> MagicMock:
            captured.append(stmt)
            return MagicMock()

        session.execute.side_effect = capture_stmt

        user_id = uuid.uuid4()
        await hard_delete_user(user_id, session)

        assert len(captured) == 1
        compiled = str(captured[0].compile(compile_kwargs={"literal_binds": False}))
        assert "users" in compiled.lower()

    async def test_idempotent_on_missing_user(self) -> None:
        """If the user row does not exist the function should not raise."""
        session = AsyncMock()
        session.add = MagicMock()
        session.execute = AsyncMock()  # no-op on non-existent row is fine

        # Should not raise even if user does not exist
        await hard_delete_user(uuid.uuid4(), session)
