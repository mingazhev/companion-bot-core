"""Risk-level and action policy for the behavior change detector.

Deterministic mappings — no model call required:
    - intent  ->  risk level
    - risk level  ->  action

Public surface:
    CONFIDENCE_THRESHOLD  — minimum score to treat a signal as a genuine intent
    CLARIFICATION_TEXT    — message sent to the user when confidence is low
    get_risk_level        — returns the RiskLevel for a given intent string
    get_action            — returns the DetectionAction for an intent + risk pair
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from tdbot.behavior.schemas import DetectionAction, RiskLevel


# Minimum confidence required to classify a message as a config-change intent.
# Signals that score below this threshold fall back to ``normal_chat``.
CONFIDENCE_THRESHOLD: Final[float] = 0.35

# Clarification text emitted when the score is above zero but below the threshold.
CLARIFICATION_TEXT: Final[str] = (
    "I'm not sure what you'd like to change. "
    "You can use /set_tone or /set_persona, "
    "or just describe what you'd like differently."
)


def get_risk_level(intent: str) -> RiskLevel:
    """Return the deterministic risk level for *intent*.

    Args:
        intent: One of the six ``DetectedIntent`` strings.

    Returns:
        ``"low"``, ``"medium"``, or ``"high"``.
    """
    match intent:
        case "persona_change":
            return "medium"
        case "safety_override_attempt":
            return "high"
        case _:
            # tone_change | skill_add_prompt | skill_remove | normal_chat
            return "low"


def get_action(intent: str, risk_level: RiskLevel) -> DetectionAction:
    """Return the recommended action for an *intent* / *risk_level* pair.

    Args:
        intent:     One of the six ``DetectedIntent`` strings.
        risk_level: The risk level previously computed by :func:`get_risk_level`.

    Returns:
        ``"auto_apply"`` for low-risk config changes,
        ``"confirm"`` for medium-risk,
        ``"refuse"`` for high-risk,
        ``"pass_through"`` for normal chat.
    """
    if intent == "normal_chat":
        return "pass_through"
    match risk_level:
        case "low":
            return "auto_apply"
        case "medium":
            return "confirm"
        case _:  # "high"
            return "refuse"
