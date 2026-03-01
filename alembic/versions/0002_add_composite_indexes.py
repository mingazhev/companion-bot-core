"""Add composite indexes for hot query paths.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-01 12:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Composite index for load_recent_messages:
    #   WHERE user_id = ? AND (ttl_expires_at IS NULL OR ttl_expires_at > now)
    #   ORDER BY created_at DESC LIMIT N
    #
    # The leading user_id column also covers standalone user_id lookups,
    # making the old single-column index redundant.
    op.execute(sa.text(
        "CREATE INDEX ix_conversation_messages_user_id_created_at "
        "ON conversation_messages (user_id, created_at DESC)"
    ))
    op.drop_index(
        "ix_conversation_messages_user_id",
        table_name="conversation_messages",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_messages_user_id_created_at",
        table_name="conversation_messages",
    )
    op.create_index(
        "ix_conversation_messages_user_id",
        "conversation_messages",
        ["user_id"],
    )
