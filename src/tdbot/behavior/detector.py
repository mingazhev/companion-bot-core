"""Behavior change detector.

Classifies an incoming user message into one of six intents and assigns
a risk level and recommended action using purely deterministic regex
heuristics (no model call).

Intent priority:
  1. ``safety_override_attempt`` — any non-zero score short-circuits all
     other intents and returns a high-risk refuse result immediately.
  2. All other intents compete by score; the highest-scoring intent wins.
  3. If the winning score is below ``CONFIDENCE_THRESHOLD`` the result
     falls back to ``normal_chat`` and a clarification question is set
     when the score is non-zero.

Public surface:
    classify  — the single entry-point for intent classification
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from tdbot.behavior.policy import (
    CLARIFICATION_TEXT,
    CONFIDENCE_THRESHOLD,
    get_action,
    get_risk_level,
)
from tdbot.behavior.schemas import DetectionResult
from tdbot.signals import Signal, compile_signals, score_signals

if TYPE_CHECKING:
    from tdbot.behavior.schemas import DetectedIntent


# ---------------------------------------------------------------------------
# Safety signals (always checked first)
# ---------------------------------------------------------------------------

_SAFETY_SIGNALS: Final[list[Signal]] = compile_signals(
    [
        # Explicit instruction / rule bypass — s? handles plurals
        (
            r"\bignore\b.{0,30}\b"
            r"(instructions?|rules?|guidelines?|policies?|restrictions?|filters?|safety)\b",
            0.8,
        ),
        (
            r"\bbypass\b.{0,30}\b"
            r"(safety|filters?|restrictions?|guardrails?|rules?)\b",
            0.8,
        ),
        (
            r"\b(disable|remove|turn off)\b.{0,25}\b"
            r"(safety|filters?|restrictions?|rules?|limits?)\b",
            0.8,
        ),
        (
            r"\boverride\b.{0,25}\b"
            r"(system|instructions?|safety|rules?|policies?)\b",
            0.8,
        ),
        # Prompt injection phrasing — handle plurals in embedded phrases
        (
            r"\bpretend\b.{0,30}\b"
            r"(no rules?|no restrictions?|unrestricted|no limits?|without restrictions?)\b",
            0.7,
        ),
        (
            r"\bforget\b.{0,20}\b"
            r"(rules?|instructions?|training|guidelines?|safety)\b",
            0.6,
        ),
        (r"\bact\b.{0,20}\b(without|freely|unfiltered|unrestricted)\b", 0.6),
        (r"\bno\b.{0,15}\b(restrictions?|filters?|rules?|safety|limits?)\b", 0.5),
        # Known jailbreak keywords
        (r"\bjailbreak\b", 0.9),
        (r"\bdan\b.{0,10}\bmode\b", 0.9),
    ],
    dotall=True,
)

# ---------------------------------------------------------------------------
# Config-change signals (scored after safety passes)
# ---------------------------------------------------------------------------

_CHANGE_SIGNALS: Final[dict[str, list[Signal]]] = {
    "persona_change": compile_signals(
        [
            (r"\byou are now\b", 0.7),
            (
                r"\bfrom now on\b.{0,20}\b"
                r"(you are|you'?re|call yourself|your name is)\b",
                0.7,
            ),
            (r"\bpretend (to be|you are|you'?re)\b", 0.6),
            (r"\broleplay (as|like)\b", 0.6),
            (r"\bact as\b.{0,30}\b(a |an )?(person|character|assistant|bot|ai)\b", 0.5),
            (r"\byour name is\b.{0,20}\bnow\b", 0.6),
            (r"\bchange your (name|identity|character|persona)\b", 0.6),
            (r"\bplay the role of\b", 0.6),
            (r"\bimagine you are\b", 0.5),
            (
                r"\bbecome\b.{0,20}\b(a |an )?"
                r"(different |new )?(person|character|assistant|entity|ai|bot)\b",
                0.5,
            ),
        ],
        dotall=True,
    ),
    "tone_change": compile_signals(
        [
            (
                r"\b(be|sound|talk|speak|respond|answer|write|communicate)\b.{0,30}\b"
                r"(more|less)\b.{0,30}\b"
                r"(formal|casual|friendly|professional|serious|playful|concise|"
                r"brief|detailed|verbose|warm|cold|direct|informal)\b",
                0.7,
            ),
            (
                r"\bchange (your |the )?(tone|style|manner|"
                r"way you (speak|talk|write|respond))\b",
                0.6,
            ),
            (
                r"\buse (simpler|more complex|technical|plain|simple|clear|"
                r"concise|elaborate) (words|language|vocabulary|terms)\b",
                0.5,
            ),
            (r"\b(adjust|modify) (your )?(communication|speaking|writing) style\b", 0.6),
            (
                r"\bstop being (so )?(formal|stiff|rigid|robotic|cold|distant|boring)\b",
                0.5,
            ),
            (
                r"\bmore (friendly|casual|formal|professional|concise|brief|"
                r"detailed|warm|relaxed|engaging|playful)\b",
                0.4,
            ),
            (r"\bless (formal|verbose|casual|cold|stiff|technical|distant)\b", 0.4),
        ],
        dotall=True,
    ),
    "skill_add_prompt": compile_signals(
        [
            (r"\badd\b.{0,25}\b(skill|capability|feature|ability|topic)\b", 0.7),
            (
                r"\bi (want|need|would like) you to (also )?"
                r"(help|assist|know about|learn|understand)\b",
                0.5,
            ),
            (
                r"\b(help|assist) me with\b.{0,40}\b"
                r"(from now on|always|going forward|in the future)\b",
                0.6,
            ),
            (
                r"\byou (should|can|could|may) also\b.{0,25}\b"
                r"(help|assist|know|support)\b",
                0.5,
            ),
            (
                r"\bplease (also )?(assist|help|support) (me )?with\b.{0,30}\b"
                r"from now on\b",
                0.5,
            ),
            (r"\blearn (about|to help with|how to help with)\b", 0.4),
        ],
        dotall=True,
    ),
    "skill_remove": compile_signals(
        [
            (r"\bremove\b.{0,25}\b(skill|capability|feature|ability|topic)\b", 0.7),
            (r"\bstop (helping|assisting) (me )?with\b", 0.7),
            (
                r"\b(disable|turn off)\b.{0,20}\b"
                r"(skill|feature|capability|assistance)\b",
                0.6,
            ),
            (r"\bdon'?t (help|assist) (me )?with\b.{0,25}\banymore\b", 0.6),
            (
                r"\bi don'?t need (your |)(help|assistance) with\b.{0,25}\banymore\b",
                0.5,
            ),
            (r"\bno (more|longer)\b.{0,20}\b(help|assistance) (with|for)\b", 0.5),
            (
                r"\bforget (that you )?(help|assist|know about|can help with)\b",
                0.4,
            ),
        ],
        dotall=True,
    ),
}


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------


def classify(text: str) -> DetectionResult:
    """Classify *text* into a behavior-change intent with risk level and action.

    Safety override signals are evaluated first and short-circuit the rest of
    the classifier if any pattern matches.  For all other intents the highest-
    scoring intent wins; ties are broken by insertion order in
    ``_CHANGE_SIGNALS``.  If the winning score is below
    :data:`~tdbot.behavior.policy.CONFIDENCE_THRESHOLD`, the result is
    ``normal_chat`` with a clarification question when score > 0.

    Args:
        text: Raw user message text (stripped or unstripped).

    Returns:
        :class:`~tdbot.behavior.schemas.DetectionResult` with all fields set.
    """
    # 1. Safety check — any match is treated as a high-risk override attempt.
    safety_score = score_signals(text, _SAFETY_SIGNALS)
    if safety_score > 0.0:
        risk = get_risk_level("safety_override_attempt")
        return DetectionResult(
            intent="safety_override_attempt",
            risk_level=risk,
            confidence=safety_score,
            action=get_action("safety_override_attempt", risk),
        )

    # 2. Score all config-change intents and pick the best.
    scores: dict[str, float] = {
        intent: score_signals(text, signals)
        for intent, signals in _CHANGE_SIGNALS.items()
    }

    best_intent_str = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent_str]

    # 3. Confidence threshold — fall back to normal chat if signal is weak.
    if best_score < CONFIDENCE_THRESHOLD:
        return DetectionResult(
            intent="normal_chat",
            risk_level="low",
            confidence=best_score,
            action="pass_through",
            clarification_question=CLARIFICATION_TEXT if best_score > 0.0 else None,
        )

    best_intent = cast("DetectedIntent", best_intent_str)
    risk = get_risk_level(best_intent)
    return DetectionResult(
        intent=best_intent,
        risk_level=risk,
        confidence=best_score,
        action=get_action(best_intent, risk),
    )
