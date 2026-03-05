"""Add mood_entries table for mood journal.

Revision ID: 0006
Revises: 0005
Create Date: 2026-03-06 00:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mood_entries",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("mood", sa.String(32), nullable=False),
        sa.Column("intensity", sa.Integer(), nullable=False),
        sa.Column("context_snippet", sa.String(50), nullable=True),
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
        "ix_mood_entries_user_id_created_at",
        "mood_entries",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("mood_entries")
