"""Parameter extraction for detected behavior change intents.

After the detector classifies a message as a config-change intent, this module
extracts concrete parameter values (tone name, persona name, skill topic) from
the raw message text using lightweight regex heuristics.

Public surface:
    extract_tone          — extract a valid tone from a tone_change message
    extract_persona_name  — extract a persona/character name from a persona_change message
    extract_skill_topic   — extract a skill topic name from a skill_add/remove message
"""

from __future__ import annotations

import re
from typing import Final

# Valid tones accepted by the system (matches handlers._VALID_TONES).
VALID_TONES: Final[frozenset[str]] = frozenset(
    {"friendly", "professional", "playful", "neutral", "casual"}
)

# Mapping from common adjective synonyms to the canonical tone name.
_TONE_ALIASES: Final[dict[str, str]] = {
    "friendly": "friendly",
    "friendlier": "friendly",
    "warm": "friendly",
    "warmer": "friendly",
    "kind": "friendly",
    "kinder": "friendly",
    "nice": "friendly",
    "nicer": "friendly",
    "welcoming": "friendly",
    "professional": "professional",
    "formal": "professional",
    "serious": "professional",
    "business": "professional",
    "corporate": "professional",
    "playful": "playful",
    "fun": "playful",
    "humorous": "playful",
    "funny": "playful",
    "funnier": "playful",
    "witty": "playful",
    "wittier": "playful",
    "lighthearted": "playful",
    "neutral": "neutral",
    "balanced": "neutral",
    "objective": "neutral",
    "casual": "casual",
    "relaxed": "casual",
    "informal": "casual",
    "chill": "casual",
    "chiller": "casual",
    "laid-back": "casual",
    "laidback": "casual",
    "conversational": "casual",
}

# Patterns to extract persona name from common phrasing.
_PERSONA_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"\byou are now\b\s+(?:called\s+)?([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\byour name is\b\s+(?:now\s+)?([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcall yourself\b\s+([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfrom now on\b.{0,30}\byou(?:'re| are)\b\s+([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpretend (?:to be|you are|you're)\b\s+(?:a |an )?([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\broleplay (?:as|like)\b\s+(?:a |an )?([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bact as\b\s+(?:a |an )?([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bplay the role of\b\s+(?:a |an )?([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbecome\b\s+(?:a |an )?(?:different |new )?([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bimagine you are\b\s+(?:a |an )?([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bchange your (?:name|identity|character|persona) to\b\s+([A-Z][a-zA-Z\-' ]{0,63})",
        re.IGNORECASE,
    ),
]

# Patterns to extract skill topic from skill_add_prompt / skill_remove messages.
_SKILL_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"\b(?:add|enable|activate)\b.{0,15}\b(?:skill|capability|feature|ability|topic)\b"
        r"(?:\s+(?:for|about|called|named))?\s+([a-zA-Z][a-zA-Z0-9 \-]{0,48})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:remove|disable|turn off|deactivate)\b.{0,15}"
        r"\b(?:skill|capability|feature|ability|topic)\b"
        r"(?:\s+(?:for|about|called|named))?\s+([a-zA-Z][a-zA-Z0-9 \-]{0,48})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:help|assist)\b\s+(?:me\s+)?with\b\s+([a-zA-Z][a-zA-Z0-9 \-]{0,48})"
        r"\s+(?:from now on|always|going forward)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bstop (?:helping|assisting)\b\s+(?:me\s+)?with\b\s+"
        r"([a-zA-Z][a-zA-Z0-9 \-]{0,48})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\blearn (?:about|to help with|how to help with)\b\s+"
        r"([a-zA-Z][a-zA-Z0-9 \-]{0,48})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdon'?t (?:help|assist)\b\s+(?:me\s+)?with\b\s+"
        r"([a-zA-Z][a-zA-Z0-9 \-]{0,48})\s+anymore",
        re.IGNORECASE,
    ),
]


def extract_tone(text: str) -> str | None:
    """Extract a valid tone from a tone_change message.

    Scans *text* for known tone names and their common aliases,
    returning the canonical tone name.  Returns ``None`` if no
    recognised tone is found.
    """
    lower = text.lower()
    for alias, canonical in _TONE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return canonical
    return None


def extract_persona_name(text: str) -> str | None:
    """Extract a persona/character name from a persona_change message.

    Applies a series of regex patterns that match common natural-language
    phrasings for persona changes and returns the captured name.
    Returns ``None`` if no name can be extracted.
    """
    for pattern in _PERSONA_PATTERNS:
        match = pattern.search(text)
        if match:
            name = match.group(1).strip().rstrip(".,!?;:")
            if name and len(name) <= 64:
                return name
    return None


def extract_skill_topic(text: str) -> str | None:
    """Extract a skill topic from a skill_add_prompt or skill_remove message.

    Returns the extracted topic string, or ``None`` if extraction fails.
    """
    for pattern in _SKILL_PATTERNS:
        match = pattern.search(text)
        if match:
            topic = match.group(1).strip().rstrip(".,!?;:")
            if topic:
                return topic
    return None
