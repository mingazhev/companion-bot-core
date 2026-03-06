"""Widen mood_entries.context_snippet from VARCHAR(50) to TEXT.

Encrypted snippets exceed 50 characters; TEXT removes the length
constraint so both encrypted and plain values fit.

Revision ID: 0009
Revises: 0008
Create Date: 2026-03-06 00:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "mood_entries",
        "context_snippet",
        existing_type=sa.String(50),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "mood_entries",
        "context_snippet",
        existing_type=sa.Text(),
        type_=sa.String(50),
        existing_nullable=True,
    )
