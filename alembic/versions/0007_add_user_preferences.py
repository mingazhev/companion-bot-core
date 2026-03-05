"""Add proactive messaging preference columns to user_profiles.

Revision ID: 0007
Revises: 0006
Create Date: 2026-03-06 00:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_profiles",
        sa.Column(
            "proactive_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "user_profiles",
        sa.Column("checkin_time", sa.Time(), nullable=True),
    )
    op.add_column(
        "user_profiles",
        sa.Column("quiet_hours_start", sa.Time(), nullable=True),
    )
    op.add_column(
        "user_profiles",
        sa.Column("quiet_hours_end", sa.Time(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_profiles", "quiet_hours_end")
    op.drop_column("user_profiles", "quiet_hours_start")
    op.drop_column("user_profiles", "checkin_time")
    op.drop_column("user_profiles", "proactive_enabled")
