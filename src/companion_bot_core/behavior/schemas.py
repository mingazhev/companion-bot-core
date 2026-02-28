"""Pydantic schemas for behavior change detection output.

Public surface:
    DetectedIntent   — literal type for the six possible intents
    RiskLevel        — literal type for low / medium / high
    DetectionAction  — literal type for the action to take
    DetectionResult  — complete output of the classifier
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Six possible intents the detector can classify.
DetectedIntent = Literal[
    "tone_change",
    "persona_change",
    "skill_add_prompt",
    "skill_remove",
    "safety_override_attempt",
    "normal_chat",
]

# Deterministic risk tier for each intent.
RiskLevel = Literal["low", "medium", "high"]

# Action the orchestrator should take based on risk.
DetectionAction = Literal[
    "auto_apply",   # low-risk: apply the change immediately
    "confirm",      # medium-risk: ask the user to confirm first
    "refuse",       # high-risk: reject the change with an explanation
    "pass_through", # normal_chat: forward the message to the model unchanged
]


class DetectionResult(BaseModel):
    """Complete output produced by the behavior change detector for one message."""

    intent: DetectedIntent = Field(
        description="Classified intent of the user message",
    )
    risk_level: RiskLevel = Field(
        description="Risk level associated with the detected intent",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score in [0.0, 1.0]; higher means stronger signal",
    )
    action: DetectionAction = Field(
        description=(
            "Recommended action: auto_apply for low-risk, confirm for medium-risk, "
            "refuse for high-risk, pass_through for normal chat"
        ),
    )
    clarification_question: str | None = Field(
        default=None,
        description=(
            "Emitted when confidence is below the threshold but above zero; "
            "prompts the user to clarify their intent"
        ),
    )
