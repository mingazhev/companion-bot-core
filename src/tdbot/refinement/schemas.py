"""Pydantic schemas for the refinement worker.

Public surface:
    RefinementRiskFlag  — risk signals reported by the refinement model
    SnapshotDelta       — proposed incremental change to the active snapshot
    RefinementResult    — full output from the non-interactive model call
"""

from __future__ import annotations

from enum import StrEnum

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
