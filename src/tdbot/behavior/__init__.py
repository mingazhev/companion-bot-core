"""Behavior change detector package.

Classifies incoming user messages into one of six intents and determines
the appropriate risk-based action using purely deterministic heuristics.

Public surface:
    classify         — classify a user message into an intent with risk/action
    DetectionResult  — complete output type from classify
    DetectedIntent   — intent literal type
    RiskLevel        — risk level literal type
    DetectionAction  — action literal type
"""

from tdbot.behavior.detector import classify
from tdbot.behavior.schemas import (
    DetectedIntent,
    DetectionAction,
    DetectionResult,
    RiskLevel,
)

__all__ = [
    "classify",
    "DetectedIntent",
    "DetectionAction",
    "DetectionResult",
    "RiskLevel",
]
