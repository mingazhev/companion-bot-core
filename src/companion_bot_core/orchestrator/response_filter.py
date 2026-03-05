"""Post-inference response filter: detect and eliminate phrase repetition.

Compares the new bot response against recent assistant messages using trigram
overlap.  When a sentence in the new response is too similar to a recent
message, it is stripped (Option B).  If the resulting text is too short or
incoherent, the caller should re-invoke inference with an anti-repetition
instruction (Option A).

Public surface:
    ngram_overlap      — compute trigram overlap ratio between two texts
    check_repetition   — identify repeated sentences in a response
    strip_repeated     — remove repeated sentences, return cleaned text
    RepetitionResult   — result of a repetition check
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return re.findall(r"[a-zа-яёA-ZА-ЯЁ0-9]+", text.lower())


def ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute n-gram overlap ratio between *text_a* and *text_b*.

    Returns a float in [0.0, 1.0] representing the fraction of overlapping
    n-grams relative to the smaller set.  Returns 0.0 if either text has
    fewer than *n* tokens.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if len(tokens_a) < n or len(tokens_b) < n:
        return 0.0

    grams_a: Counter[tuple[str, ...]] = Counter(
        tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)
    )
    grams_b: Counter[tuple[str, ...]] = Counter(
        tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1)
    )

    if not grams_a or not grams_b:
        return 0.0

    intersection = sum((grams_a & grams_b).values())
    smaller = min(sum(grams_a.values()), sum(grams_b.values()))
    return intersection / smaller if smaller > 0 else 0.0


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences on common sentence-ending punctuation."""
    raw = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


class RepetitionResult(NamedTuple):
    """Result of a repetition check."""

    repeated_phrases: list[str]
    cleaned_text: str


def check_repetition(
    response: str,
    recent_messages: list[str],
    *,
    threshold: float = 0.6,
    min_sentence_tokens: int = 4,
) -> RepetitionResult:
    """Check *response* for sentences that overlap with *recent_messages*.

    Args:
        response:            The new bot response text.
        recent_messages:     List of recent assistant message texts (newest first).
        threshold:           Trigram overlap ratio above which a sentence is
                             considered repeated.
        min_sentence_tokens: Sentences shorter than this are skipped (greetings,
                             interjections are rarely problematic repetitions).

    Returns:
        A :class:`RepetitionResult` with the list of repeated phrases and a
        cleaned response with those phrases removed.
    """
    sentences = _split_sentences(response)
    repeated: list[str] = []
    kept: list[str] = []

    for sentence in sentences:
        tokens = _tokenize(sentence)
        if len(tokens) < min_sentence_tokens:
            kept.append(sentence)
            continue

        is_repeated = False
        for prev in recent_messages:
            overlap = ngram_overlap(sentence, prev)
            if overlap >= threshold:
                is_repeated = True
                repeated.append(sentence)
                break

        if not is_repeated:
            kept.append(sentence)

    cleaned = " ".join(kept).strip()
    return RepetitionResult(repeated_phrases=repeated, cleaned_text=cleaned)


def build_anti_repetition_instruction(repeated_phrases: list[str]) -> str:
    """Build an instruction telling the model to avoid repeated phrases.

    Used as a fallback (Option A) when stripping leaves the response too short.
    """
    joined = "; ".join(repeated_phrases[:3])
    return f"Ты уже говорил: '{joined}'. Не повторяй эти мысли, скажи что-то новое."
