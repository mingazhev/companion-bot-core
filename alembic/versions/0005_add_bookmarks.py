"""Add bookmarks table for conversation bookmarks.

Revision ID: 0005
Revises: 0004
Create Date: 2026-03-05 00:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "bookmarks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("bot_response", sa.Text(), nullable=False),
        sa.Column("tag", sa.String(128), nullable=True),
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
        "ix_bookmarks_user_id_created_at",
        "bookmarks",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("bookmarks")
