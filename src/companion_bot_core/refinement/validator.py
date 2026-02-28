"""Policy validation for refinement outputs.

Checks that a ``RefinementResult`` is safe to apply before creating a new
prompt snapshot.  Validation covers two layers:

1. Model-reported risk flags (prompt_injection, unsafe_role_change, …).
2. Static text scan of proposed delta fields for injection patterns.

Public surface:
    validate_refinement_result(result) -> list[str]
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from companion_bot_core.prompt.merge_builder import SECTION_SEP
from companion_bot_core.refinement.schemas import RefinementRiskFlag

if TYPE_CHECKING:
    from companion_bot_core.refinement.schemas import RefinementResult

# Patterns that indicate obvious prompt-injection attempts in proposed text.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"ignore\s+(previous|above|all)\s+(instructions?|rules?|prompts?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(all\s+)?(previous\s+)?(instructions?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(r"you\s+are\s+now\s+(?!friendly|helpful)", re.IGNORECASE),
    re.compile(
        r"(forget|override)\s+(your|all|previous)\s+(instructions?|rules?|guidelines?)",
        re.IGNORECASE,
    ),
]

# Human-readable messages for each model-reported risk flag.
_FLAG_MESSAGES: dict[RefinementRiskFlag, str] = {
    RefinementRiskFlag.PROMPT_INJECTION: (
        "model detected prompt-injection attempt in context"
    ),
    RefinementRiskFlag.UNSAFE_ROLE_CHANGE: (
        "proposed delta contains unsafe role change"
    ),
    RefinementRiskFlag.POLICY_VIOLATION: "proposed delta violates safety policy",
    RefinementRiskFlag.SCHEMA_VIOLATION: "proposed delta schema is invalid",
}


def _contains_injection_pattern(text: str) -> bool:
    """Return True if *text* matches any known prompt-injection pattern."""
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def validate_refinement_result(result: RefinementResult) -> list[str]:
    """Return a list of policy violation messages (empty list = safe to apply).

    Checks:
    1. Model-reported risk flags.
    2. Static injection-pattern scan of each proposed delta text field.

    Args:
        result: Validated ``RefinementResult`` from the model.

    Returns:
        List of human-readable violation descriptions.  An empty list means
        all policy checks passed and the result is safe to apply.
    """
    violations: list[str] = []

    # --- model-reported risk flags ---
    for flag in result.risk_flags:
        message = _FLAG_MESSAGES.get(flag)
        if message:
            violations.append(message)

    # --- static text scan of proposed delta ---
    delta = result.proposed_delta
    candidate_texts: list[str] = []
    if delta.persona_segment:
        candidate_texts.append(delta.persona_segment)
    if delta.long_term_profile:
        candidate_texts.append(delta.long_term_profile)
    if delta.skill_packs:
        candidate_texts.extend(delta.skill_packs.values())

    for text in candidate_texts:
        if _contains_injection_pattern(text):
            violations.append("proposed text matches prompt-injection pattern")
            break  # one violation is enough to reject the whole result

    # --- prompt-structure integrity: reject deltas containing the section separator ---
    for text in candidate_texts:
        if SECTION_SEP in text:
            violations.append("proposed text contains prompt section separator")
            break

    return violations
