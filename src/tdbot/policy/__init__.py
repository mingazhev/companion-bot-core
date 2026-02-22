"""Policy guardrail layer for TdBot.

Provides deterministic, model-free checks that run before message processing:

- ``check_prompt_injection``   — detect injection attempts in raw user text
- ``check_unsafe_role_change`` — detect privilege-escalation phrasing
- ``check_risky_capability``   — detect requests for out-of-scope capabilities
- ``is_user_abuse_blocked``    — Redis-backed abuse throttle
- ``record_policy_violation``  — increment per-user violation counter

All checks return a :class:`~tdbot.policy.schemas.GuardrailResult` with
``allowed=False`` and a reason string when the guardrail fires.
"""

from __future__ import annotations

from tdbot.policy.guardrails import (
    check_prompt_injection,
    check_risky_capability,
    check_unsafe_role_change,
)

__all__ = [
    "check_prompt_injection",
    "check_risky_capability",
    "check_unsafe_role_change",
]
