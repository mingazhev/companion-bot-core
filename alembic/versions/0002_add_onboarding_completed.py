"""Add onboarding_completed column to users table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-28 00:00:00.000000

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
    op.add_column(
        "users",
        sa.Column(
            "onboarding_completed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
            comment="True once the user has answered the initial onboarding question",
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "onboarding_completed")
