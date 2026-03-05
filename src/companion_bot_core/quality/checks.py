"""Deterministic response quality checks — no LLM required.

Reusable functions for validating bot response patterns.  Used by CI tests
and the repetition guard (``orchestrator.response_filter``).

Public surface:
    has_ai_markers       — detect common AI-assistant phrases
    count_bullet_points  — count markdown/numbered list items
    split_sentences      — split text into sentences
    count_sentences      — count sentences in text
    has_menu_pattern     — detect numbered/bulleted menus (3+ items)
    is_short_farewell    — check if text is a brief goodbye
    contains_name        — check if a name appears in text
    ngram_overlap        — compute n-gram overlap ratio between two texts
"""

from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------------------
# AI markers — common robotic / assistant phrases
# ---------------------------------------------------------------------------

_AI_MARKER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"как\s+(языковая\s+модель|ИИ|искусственный\s+интеллект)",
        r"я\s+не\s+могу\s+испытывать\s+эмоции",
        r"я\s+(?:всего\s+лишь\s+)?(?:чат-?бот|бот|программа)",
        r"как\s+AI\b",
        r"as\s+an?\s+(?:AI|language\s+model)",
        r"I\s+(?:don'?t|cannot)\s+(?:have|experience)\s+(?:feelings|emotions)",
        r"I'?m\s+(?:just\s+)?(?:a\s+)?(?:chat)?bot",
        r"рад\s+(?:был\s+)?помочь",
        r"чем\s+(?:ещё|еще)\s+(?:могу|я\s+могу)\s+помочь",
        r"обратитесь\s+к\s+(?:специалисту|врачу|психологу)",
    ]
]


def has_ai_markers(text: str) -> list[str]:
    """Return list of AI-marker phrases found in *text*."""
    return [m.group() for p in _AI_MARKER_PATTERNS if (m := p.search(text))]


# ---------------------------------------------------------------------------
# Bullet points and menu patterns
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*[-•*]\s+\S", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)


def count_bullet_points(text: str) -> int:
    """Count markdown-style bullet points (``-``, ``*``, ``•``) in *text*."""
    return len(_BULLET_RE.findall(text))


def _count_numbered_items(text: str) -> int:
    """Count numbered list items (``1)`` or ``1.``) in *text*."""
    return len(_NUMBERED_RE.findall(text))


def has_menu_pattern(text: str) -> bool:
    """Detect a "menu" pattern — 3+ bulleted or numbered items."""
    return count_bullet_points(text) >= 3 or _count_numbered_items(text) >= 3  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Sentence utilities
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences on sentence-ending punctuation."""
    stripped = text.strip()
    if not stripped:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(stripped) if p.strip()]


def count_sentences(text: str) -> int:
    """Count sentences in *text* by splitting on sentence-ending punctuation."""
    return len(split_sentences(text))


def is_short_farewell(text: str, max_sentences: int = 3) -> bool:
    """Return ``True`` if *text* looks like a short farewell (<=*max_sentences*).

    Checks both sentence count and the presence of farewell keywords.
    """
    farewell_re = re.compile(
        r"(?:пока|до\s+свидания|до\s+встречи|удачи|спокойной\s+ночи"
        r"|хорошего\s+(?:дня|вечера)|bye|goodbye|see\s+you|good\s+night)",
        re.IGNORECASE,
    )
    return count_sentences(text) <= max_sentences and bool(farewell_re.search(text))


# ---------------------------------------------------------------------------
# Name detection
# ---------------------------------------------------------------------------


def contains_name(text: str, name: str) -> bool:
    """Return ``True`` if *name* appears in *text* (case-insensitive, word boundary)."""
    if not name:
        return False
    pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    return bool(pattern.search(text))


# ---------------------------------------------------------------------------
# N-gram overlap (canonical implementation — reused by response_filter)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zа-яёA-ZА-ЯЁ0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return _TOKEN_RE.findall(text.lower())


def ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute n-gram overlap ratio between *text_a* and *text_b*.

    Returns a float in [0.0, 1.0] representing the fraction of overlapping
    n-grams relative to the smaller set.  Returns 0.0 if either text has
    fewer than *n* tokens.
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if len(tokens_a) < n or len(tokens_b) < n:
        return 0.0

    grams_a: Counter[tuple[str, ...]] = Counter(
        tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)
    )
    grams_b: Counter[tuple[str, ...]] = Counter(
        tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1)
    )

    intersection = sum((grams_a & grams_b).values())
    smaller = min(sum(grams_a.values()), sum(grams_b.values()))
    return intersection / smaller if smaller > 0 else 0.0
