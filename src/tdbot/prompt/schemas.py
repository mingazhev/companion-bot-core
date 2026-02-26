"""Pydantic schemas for prompt state: components, snapshot records, and metadata.

Public surface:
    SnapshotSource   — literal type for snapshot creation reasons
    PromptComponents — all segments merged to build the system prompt
    SnapshotRecord   — a single immutable prompt snapshot (in-memory repr)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# Immutable set of reasons that can create a new snapshot.
SnapshotSource = Literal["initial", "user_command", "behavior_change", "refinement", "rollback"]


class PromptComponents(BaseModel):
    """All segments merged in order to produce the compiled system prompt."""

    base_system_template: str = Field(
        description="Core identity rules and safety policy (shared across all users)",
    )
    persona_segment: str = Field(
        default="",
        description="Per-user persona overrides (name, tone, style constraints)",
    )
    skill_packs: dict[str, str] = Field(
        default_factory=dict,
        description="Map of skill_name -> skill system-prompt fragment",
    )
    short_term_window: str = Field(
        default="",
        description="Summarised view of the recent conversation turn window",
    )
    long_term_profile: str = Field(
        default="",
        description="Compacted long-term user profile produced by memory compaction",
    )


class SnapshotRecord(BaseModel):
    """In-memory representation of a single immutable prompt snapshot.

    Mirrors the ``prompt_snapshots`` ORM model but is decoupled from SQLAlchemy
    so that the prompt package can be used and tested without a database.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    version: int = Field(ge=1, description="Monotonically increasing per-user version")
    system_prompt: str = Field(description="The fully compiled system prompt text")
    skill_prompts_json: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw skill prompts used to produce this snapshot (for audit)",
    )
    source: SnapshotSource = Field(description="What triggered this snapshot creation")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
    )
