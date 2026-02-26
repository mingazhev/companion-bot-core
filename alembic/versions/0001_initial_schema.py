"""Initial database schema.

Revision ID: 0001
Revises:
Create Date: 2026-02-22 00:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("locale", sa.String(length=16), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("telegram_user_id"),
    )

    # ------------------------------------------------------------------
    # user_profiles
    # ------------------------------------------------------------------
    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("persona_name", sa.Text(), nullable=True),
        sa.Column("tone", sa.Text(), nullable=True),
        sa.Column("style_constraints", sa.Text(), nullable=True),
        sa.Column(
            "safety_level",
            sa.String(length=32),
            nullable=False,
            server_default="moderate",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )

    # ------------------------------------------------------------------
    # prompt_snapshots
    # ------------------------------------------------------------------
    op.create_table(
        "prompt_snapshots",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column(
            "skill_prompts_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_prompt_snapshots_user_id",
        "prompt_snapshots",
        ["user_id"],
    )

    # ------------------------------------------------------------------
    # conversation_messages
    # ------------------------------------------------------------------
    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ttl_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversation_messages_user_id",
        "conversation_messages",
        ["user_id"],
    )
    op.create_index(
        "ix_conversation_messages_created_at",
        "conversation_messages",
        ["created_at"],
    )
    op.create_index(
        "ix_conversation_messages_ttl_expires_at",
        "conversation_messages",
        ["ttl_expires_at"],
    )

    # ------------------------------------------------------------------
    # memory_compactions
    # ------------------------------------------------------------------
    op.create_table(
        "memory_compactions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column(
            "triggered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("snapshot_id_before", sa.UUID(), nullable=True),
        sa.Column("snapshot_id_after", sa.UUID(), nullable=True),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["snapshot_id_before"], ["prompt_snapshots.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id_after"], ["prompt_snapshots.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_memory_compactions_user_id",
        "memory_compactions",
        ["user_id"],
    )

    # ------------------------------------------------------------------
    # behavior_change_events
    # ------------------------------------------------------------------
    op.create_table(
        "behavior_change_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("intent", sa.String(length=64), nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "applied", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "confirmed", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("source_message_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_behavior_change_events_user_id",
        "behavior_change_events",
        ["user_id"],
    )

    # ------------------------------------------------------------------
    # jobs
    # ------------------------------------------------------------------
    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "payload_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])

    # ------------------------------------------------------------------
    # audit_log
    # ------------------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column(
            "event_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "details_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_log_user_id", "audit_log", ["user_id"])
    op.create_index("ix_audit_log_event_type", "audit_log", ["event_type"])
    op.create_index("ix_audit_log_event_at", "audit_log", ["event_at"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("jobs")
    op.drop_table("behavior_change_events")
    op.drop_table("memory_compactions")
    op.drop_table("conversation_messages")
    op.drop_table("prompt_snapshots")
    op.drop_table("user_profiles")
    op.drop_table("users")
