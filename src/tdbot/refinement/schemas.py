"""Pydantic schemas for the refinement worker.

Public surface:
    RefinementRiskFlag  — risk signals reported by the refinement model
    SnapshotDelta       — proposed incremental change to the active snapshot
    RefinementResult    — full output from the non-interactive model call
    RefinementJob       — envelope for a dequeued Redis job item
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RefinementRiskFlag(StrEnum):
    """Risk signals that the refinement model may raise."""

    PROMPT_INJECTION = "prompt_injection"
    UNSAFE_ROLE_CHANGE = "unsafe_role_change"
    POLICY_VIOLATION = "policy_violation"
    SCHEMA_VIOLATION = "schema_violation"


class SnapshotDelta(BaseModel):
    """Proposed incremental changes to the active prompt snapshot.

    A ``None`` value for any field means "no change requested" — the existing
    value will be carried over when building the new snapshot.
    """

    persona_segment: str | None = Field(
        default=None,
        description="Updated persona text; None means no change",
    )
    skill_packs: dict[str, str] | None = Field(
        default=None,
        description="Replacement skill packs map; None means no change",
    )
    long_term_profile: str | None = Field(
        default=None,
        description="Updated long-term user profile summary; None means no change",
    )


class RefinementResult(BaseModel):
    """Validated output returned by a non-interactive refinement model call."""

    proposed_delta: SnapshotDelta
    rationale: str = Field(description="Model explanation for the proposed changes")
    risk_flags: list[RefinementRiskFlag] = Field(
        default_factory=list,
        description="Risk signals raised by the model",
    )


class RefinementJob(BaseModel):
    """Envelope for a refinement job item dequeued from Redis."""

    user_id: uuid.UUID
    job_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    attempt: int = Field(default=0, ge=0)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Original queue payload fields (trigger, count, etc.)",
    )
