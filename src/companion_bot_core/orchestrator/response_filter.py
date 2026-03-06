"""Post-inference response filter: detect and eliminate phrase repetition.

Compares the new bot response against recent assistant messages using trigram
overlap.  When a sentence in the new response is too similar to a recent
message, it is stripped (Option B).  If the resulting text is too short or
incoherent, the caller should re-invoke inference with an anti-repetition
instruction (Option A).

Public surface:
    check_repetition   — identify repeated sentences in a response
    RepetitionResult   — result of a repetition check
"""

from __future__ import annotations

from typing import NamedTuple

from companion_bot_core.quality.checks import ngram_overlap, split_sentences, tokenize


class RepetitionResult(NamedTuple):
    """Result of a repetition check."""

    repeated_phrases: list[str]
    cleaned_text: str


def _extract_repeated_subphrases(
    sentence_tokens: list[str],
    prev_tokens_sets: list[set[tuple[str, ...]]],
    *,
    window: int = 5,
    min_matches: int = 2,
) -> bool:
    """Check if a sentence contains a repeated sub-phrase.

    Extracts sliding windows of *window* tokens from the sentence and checks
    whether enough of them appear in any of the previous token gram sets.
    Returns True if the sentence contains a repeated sub-phrase found in at
    least *min_matches* previous messages.
    """
    if len(sentence_tokens) < window:
        return False

    sentence_grams = {
        tuple(sentence_tokens[i : i + window])
        for i in range(len(sentence_tokens) - window + 1)
    }

    match_count = 0
    for prev_grams in prev_tokens_sets:
        shared = sentence_grams & prev_grams
        # If 2+ windows from this sentence overlap, the sub-phrase is repeated
        if len(shared) >= 2:  # noqa: PLR2004
            match_count += 1
            if match_count >= min_matches:
                return True
    return False


def check_repetition(
    response: str,
    recent_messages: list[str],
    *,
    threshold: float = 0.5,
    min_sentence_tokens: int = 4,
) -> RepetitionResult:
    """Check *response* for sentences that overlap with *recent_messages*.

    Args:
        response:            The new bot response text.
        recent_messages:     List of recent assistant message texts (newest first).
        threshold:           Trigram overlap ratio above which a sentence is
                             considered repeated.  Lowered from 0.6 to 0.5 to
                             catch near-paraphrase repetitions (e.g. the same
                             reassurance phrased slightly differently).
        min_sentence_tokens: Sentences shorter than this are skipped (greetings,
                             interjections are rarely problematic repetitions).

    Returns:
        A :class:`RepetitionResult` with the list of repeated phrases and a
        cleaned response with those phrases removed.
    """
    sentences = split_sentences(response)
    repeated: list[str] = []
    kept: list[str] = []

    # Pre-split previous messages into sentences for finer-grained comparison.
    # Comparing new sentences against individual previous sentences catches
    # short repeated phrases that get diluted in full-message comparison.
    prev_sentences: list[str] = []
    for prev in recent_messages:
        prev_sentences.extend(split_sentences(prev))

    # Build 5-gram sets per previous message for sub-phrase detection.
    # This catches phrases like "не делает тебя плохой мамой" repeated
    # across multiple messages even when embedded in different sentences.
    _subphrase_window = 5
    prev_gram_sets: list[set[tuple[str, ...]]] = []
    for prev in recent_messages:
        toks = tokenize(prev)
        if len(toks) >= _subphrase_window:
            prev_gram_sets.append({
                tuple(toks[i : i + _subphrase_window])
                for i in range(len(toks) - _subphrase_window + 1)
            })

    for sentence in sentences:
        tokens = tokenize(sentence)
        if len(tokens) < min_sentence_tokens:
            kept.append(sentence)
            continue

        is_repeated = False
        # Check against individual sentences from previous messages.
        for prev_sent in prev_sentences:
            if len(tokenize(prev_sent)) < min_sentence_tokens:
                continue
            overlap = ngram_overlap(sentence, prev_sent)
            if overlap >= threshold:
                is_repeated = True
                repeated.append(sentence)
                break

        if not is_repeated:
            # Fallback: also check against whole previous messages.
            for prev in recent_messages:
                overlap = ngram_overlap(sentence, prev)
                if overlap >= threshold:
                    is_repeated = True
                    repeated.append(sentence)
                    break

        if (
            not is_repeated
            and prev_gram_sets
            and _extract_repeated_subphrases(
                tokens, prev_gram_sets,
                window=_subphrase_window, min_matches=2,
            )
        ):
            # Sub-phrase check: detect when a distinctive phrase appears
            # in 2+ previous messages (e.g. same reassurance repeated).
            is_repeated = True
            repeated.append(sentence)

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
