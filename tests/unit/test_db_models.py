"""Unit tests for SQLAlchemy ORM model definitions.

These tests verify the schema structure without requiring a live database
connection — they introspect the SQLAlchemy metadata only.
"""

from __future__ import annotations

import uuid

from tdbot.db.models import (
    AuditLog,
    Base,
    BehaviorChangeEvent,
    ConversationMessage,
    Job,
    PromptSnapshot,
    User,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(name: str) -> object:
    """Return the SQLAlchemy Table object for the given name."""
    return Base.metadata.tables[name]


def _col_names(table_name: str) -> set[str]:
    return {col.name for col in Base.metadata.tables[table_name].columns}


def _fk_targets(table_name: str) -> set[str]:
    """Return the set of target table.column strings for all FKs in a table."""
    return {
        fk.target_fullname
        for col in Base.metadata.tables[table_name].columns
        for fk in col.foreign_keys
    }


# ---------------------------------------------------------------------------
# Metadata registration
# ---------------------------------------------------------------------------


class TestTablesRegistered:
    expected_tables = {
        "users",
        "user_profiles",
        "prompt_snapshots",
        "conversation_messages",
        "memory_compactions",
        "behavior_change_events",
        "jobs",
        "audit_log",
    }

    def test_all_tables_in_metadata(self) -> None:
        assert self.expected_tables.issubset(set(Base.metadata.tables.keys()))


# ---------------------------------------------------------------------------
# users table
# ---------------------------------------------------------------------------


class TestUsersTable:
    def test_required_columns_present(self) -> None:
        cols = _col_names("users")
        for name in ("id", "telegram_user_id", "created_at", "status", "locale", "timezone"):
            assert name in cols, f"Missing column: {name}"

    def test_primary_key_is_id(self) -> None:
        table = Base.metadata.tables["users"]
        pk_cols = [col.name for col in table.primary_key.columns]
        assert pk_cols == ["id"]

    def test_telegram_user_id_is_unique(self) -> None:
        table = Base.metadata.tables["users"]
        unique_cols: set[str] = set()
        for constraint in table.constraints:
            if hasattr(constraint, "columns"):
                cols = [c.name for c in constraint.columns]
                if cols == ["telegram_user_id"]:
                    unique_cols.update(cols)
        assert "telegram_user_id" in unique_cols

    def test_orm_default_uuid(self) -> None:
        user = User(telegram_user_id=12345, status="active")
        assert user.id is None or isinstance(user.id, uuid.UUID)


# ---------------------------------------------------------------------------
# user_profiles table
# ---------------------------------------------------------------------------


class TestUserProfilesTable:
    def test_required_columns(self) -> None:
        cols = _col_names("user_profiles")
        expected = (
            "user_id", "persona_name", "tone", "style_constraints", "safety_level", "updated_at"
        )
        for name in expected:
            assert name in cols

    def test_fk_to_users(self) -> None:
        assert "users.id" in _fk_targets("user_profiles")

    def test_primary_key_is_user_id(self) -> None:
        table = Base.metadata.tables["user_profiles"]
        pk_cols = [col.name for col in table.primary_key.columns]
        assert pk_cols == ["user_id"]


# ---------------------------------------------------------------------------
# prompt_snapshots table
# ---------------------------------------------------------------------------


class TestPromptSnapshotsTable:
    def test_required_columns(self) -> None:
        cols = _col_names("prompt_snapshots")
        for name in (
            "id",
            "user_id",
            "version",
            "system_prompt",
            "skill_prompts_json",
            "source",
            "created_at",
        ):
            assert name in cols

    def test_fk_to_users(self) -> None:
        assert "users.id" in _fk_targets("prompt_snapshots")

    def test_user_id_index_exists(self) -> None:
        table = Base.metadata.tables["prompt_snapshots"]
        index_cols = {
            col.name
            for idx in table.indexes
            for col in idx.columns
        }
        assert "user_id" in index_cols

    def test_orm_instantiation(self) -> None:
        snap = PromptSnapshot(
            user_id=uuid.uuid4(),
            version=1,
            system_prompt="You are a helpful assistant.",
            skill_prompts_json={},
            source="initial",
        )
        assert snap.version == 1
        assert snap.source == "initial"


# ---------------------------------------------------------------------------
# conversation_messages table
# ---------------------------------------------------------------------------


class TestConversationMessagesTable:
    def test_required_columns(self) -> None:
        cols = _col_names("conversation_messages")
        for name in (
            "id",
            "user_id",
            "role",
            "content",
            "created_at",
            "ttl_expires_at",
        ):
            assert name in cols

    def test_ttl_column_is_nullable(self) -> None:
        table = Base.metadata.tables["conversation_messages"]
        col = table.c["ttl_expires_at"]
        assert col.nullable

    def test_fk_to_users(self) -> None:
        assert "users.id" in _fk_targets("conversation_messages")

    def test_ttl_index_exists(self) -> None:
        table = Base.metadata.tables["conversation_messages"]
        index_cols = {
            col.name
            for idx in table.indexes
            for col in idx.columns
        }
        assert "ttl_expires_at" in index_cols

    def test_orm_instantiation(self) -> None:
        msg = ConversationMessage(
            user_id=uuid.uuid4(),
            role="user",
            content="Hello!",
        )
        assert msg.role == "user"
        assert msg.ttl_expires_at is None


# ---------------------------------------------------------------------------
# memory_compactions table
# ---------------------------------------------------------------------------


class TestMemoryCompactionsTable:
    def test_required_columns(self) -> None:
        cols = _col_names("memory_compactions")
        for name in (
            "id",
            "user_id",
            "triggered_at",
            "snapshot_id_before",
            "snapshot_id_after",
            "message_count",
            "status",
        ):
            assert name in cols

    def test_fks(self) -> None:
        targets = _fk_targets("memory_compactions")
        assert "users.id" in targets
        assert "prompt_snapshots.id" in targets


# ---------------------------------------------------------------------------
# behavior_change_events table
# ---------------------------------------------------------------------------


class TestBehaviorChangeEventsTable:
    def test_required_columns(self) -> None:
        cols = _col_names("behavior_change_events")
        for name in (
            "id",
            "user_id",
            "detected_at",
            "intent",
            "risk_level",
            "confidence",
            "applied",
            "confirmed",
            "source_message_id",
        ):
            assert name in cols

    def test_fk_to_users(self) -> None:
        assert "users.id" in _fk_targets("behavior_change_events")

    def test_column_defaults_configured(self) -> None:
        table = Base.metadata.tables["behavior_change_events"]
        # The columns have server_default or default configured.
        applied_col = table.c["applied"]
        confirmed_col = table.c["confirmed"]
        assert applied_col.server_default is not None or applied_col.default is not None
        assert confirmed_col.server_default is not None or confirmed_col.default is not None

    def test_orm_explicit_values(self) -> None:
        event = BehaviorChangeEvent(
            user_id=uuid.uuid4(),
            intent="tone_change",
            risk_level="low",
            confidence=0.9,
            applied=False,
            confirmed=False,
        )
        assert event.applied is False
        assert event.confirmed is False


# ---------------------------------------------------------------------------
# jobs table
# ---------------------------------------------------------------------------


class TestJobsTable:
    def test_required_columns(self) -> None:
        cols = _col_names("jobs")
        for name in (
            "id",
            "type",
            "user_id",
            "status",
            "payload_json",
            "created_at",
            "started_at",
            "finished_at",
            "attempt",
            "error",
        ):
            assert name in cols

    def test_fk_to_users(self) -> None:
        assert "users.id" in _fk_targets("jobs")

    def test_nullable_timestamps(self) -> None:
        table = Base.metadata.tables["jobs"]
        assert table.c["started_at"].nullable
        assert table.c["finished_at"].nullable
        assert table.c["error"].nullable

    def test_orm_instantiation(self) -> None:
        job = Job(
            type="refinement",
            user_id=uuid.uuid4(),
            status="pending",
            payload_json={"snapshot_id": str(uuid.uuid4())},
            attempt=0,
        )
        assert job.attempt == 0
        assert job.error is None


# ---------------------------------------------------------------------------
# audit_log table
# ---------------------------------------------------------------------------


class TestAuditLogTable:
    def test_required_columns(self) -> None:
        cols = _col_names("audit_log")
        for name in ("id", "user_id", "event_type", "event_at", "details_json"):
            assert name in cols

    def test_user_id_is_nullable(self) -> None:
        table = Base.metadata.tables["audit_log"]
        assert table.c["user_id"].nullable

    def test_fk_to_users(self) -> None:
        assert "users.id" in _fk_targets("audit_log")

    def test_orm_instantiation(self) -> None:
        entry = AuditLog(
            event_type="user_registered",
            details_json={"action": "registered"},
        )
        assert entry.user_id is None
        assert entry.event_type == "user_registered"


# ---------------------------------------------------------------------------
# Relationship smoke tests (no DB needed — just verify the mapper is wired)
# ---------------------------------------------------------------------------


class TestRelationships:
    def test_user_has_profile_relationship(self) -> None:
        from sqlalchemy import inspect
        mapper = inspect(User)
        assert "profile" in {r.key for r in mapper.relationships}

    def test_user_has_prompt_snapshots_relationship(self) -> None:
        from sqlalchemy import inspect
        mapper = inspect(User)
        assert "prompt_snapshots" in {r.key for r in mapper.relationships}

    def test_user_has_jobs_relationship(self) -> None:
        from sqlalchemy import inspect
        mapper = inspect(User)
        assert "jobs" in {r.key for r in mapper.relationships}

    def test_user_profile_back_populates_user(self) -> None:
        from sqlalchemy import inspect
        mapper = inspect(UserProfile)
        assert "user" in {r.key for r in mapper.relationships}
