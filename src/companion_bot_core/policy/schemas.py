"""Pydantic schemas for the policy guardrail layer.

Public surface:
    GuardrailViolation — literal type for recognised violation categories
    GuardrailResult    — outcome of a single guardrail check
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Violation categories surfaced by the guardrail layer.
GuardrailViolation = Literal[
    "prompt_injection",    # User attempts to inject instructions into the system prompt
    "unsafe_role_change",  # User attempts to assume a privileged or system role
    "risky_capability",    # User requests a capability that is out-of-scope for v1
    "abuse_throttle",      # User exceeded the per-user policy-violation threshold
]


class GuardrailResult(BaseModel):
    """Outcome of a single guardrail check."""

    allowed: bool = Field(
        description="True if the message passes the guardrail; False if it is blocked",
    )
    violation: GuardrailViolation | None = Field(
        default=None,
        description="The violation category if allowed is False; None otherwise",
    )
    reason: str | None = Field(
        default=None,
        description="Human-readable explanation of why the message was blocked",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Detection confidence in [0.0, 1.0]; 0.0 for passing messages",
    )
