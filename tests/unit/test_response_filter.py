"""Unit tests for companion_bot_core.orchestrator.response_filter — repetition guard."""

from __future__ import annotations

import pytest

from companion_bot_core.orchestrator.response_filter import (
    build_anti_repetition_instruction,
    check_repetition,
)
from companion_bot_core.quality.checks import ngram_overlap

# ---------------------------------------------------------------------------
# ngram_overlap
# ---------------------------------------------------------------------------


class TestNgramOverlap:
    def test_identical_texts_return_1(self) -> None:
        text = "Это нормально хотеть сбежать от рутины"
        assert ngram_overlap(text, text) == pytest.approx(1.0)

    def test_completely_different_texts_return_0(self) -> None:
        a = "Сегодня прекрасная погода за окном"
        b = "Мне нравится читать книги вечером"
        assert ngram_overlap(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        a = "Это нормально хотеть сбежать от рутины и проблем"
        b = "Это нормально хотеть сбежать. Все так делают иногда"
        overlap = ngram_overlap(a, b)
        assert 0.3 < overlap < 1.0  # noqa: PLR2004

    def test_short_text_below_n_returns_0(self) -> None:
        assert ngram_overlap("да", "нет", n=3) == 0.0

    def test_empty_text_returns_0(self) -> None:
        assert ngram_overlap("", "привет мир как дела") == 0.0
        assert ngram_overlap("привет мир как дела", "") == 0.0

    def test_case_insensitive(self) -> None:
        a = "Привет Мир Как Дела Сегодня"
        b = "привет мир как дела сегодня"
        assert ngram_overlap(a, b) == pytest.approx(1.0)

    def test_custom_n(self) -> None:
        a = "один два три четыре пять шесть"
        b = "один два три четыре пять шесть"
        assert ngram_overlap(a, b, n=2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# check_repetition — overlap detection
# ---------------------------------------------------------------------------


class TestCheckRepetition:
    def test_detects_repeated_sentence(self) -> None:
        response = "Я понимаю тебя. Это нормально хотеть сбежать от рутины. Давай поговорим."
        recent = ["Это нормально хотеть сбежать от рутины и забот."]
        result = check_repetition(response, recent)
        assert len(result.repeated_phrases) >= 1
        assert "сбежать" in result.repeated_phrases[0].lower()

    def test_no_repetition_clean_pass(self) -> None:
        response = "Расскажи мне подробнее. Что именно тебя беспокоит?"
        recent = ["Привет! Рада тебя видеть."]
        result = check_repetition(response, recent)
        assert result.repeated_phrases == []
        assert result.cleaned_text == response

    def test_stripped_response_preserves_non_repeated(self) -> None:
        response = "Я тут для тебя. Это нормально хотеть сбежать от рутины. Как прошёл день?"
        recent = ["Это нормально хотеть сбежать от рутины и проблем."]
        result = check_repetition(response, recent)
        assert "Я тут для тебя" in result.cleaned_text
        assert "Как прошёл день" in result.cleaned_text

    def test_short_sentences_not_flagged(self) -> None:
        """Greetings and short interjections should not be treated as repetition."""
        response = "Привет! Как дела? Расскажи что нового."
        recent = ["Привет! Как ты?"]
        result = check_repetition(response, recent, min_sentence_tokens=4)
        # "Привет!" has only 1 token — should be kept
        assert result.repeated_phrases == []

    def test_multiple_recent_messages(self) -> None:
        response = "Понимаю твоё раздражение. Все устают от рутины."
        recent = [
            "Все устают от рутины и забот повседневных.",
            "Расскажи что случилось на работе.",
        ]
        result = check_repetition(response, recent)
        assert len(result.repeated_phrases) >= 1

    def test_threshold_boundary(self) -> None:
        """Overlap just below threshold should not be flagged."""
        response = "Сегодня мне хочется поговорить о чём-то новом."
        recent = ["Вчера мы говорили о многом новом и интересном."]
        result = check_repetition(response, recent, threshold=0.9)
        assert result.repeated_phrases == []

    def test_empty_response(self) -> None:
        result = check_repetition("", ["Some recent message"])
        assert result.repeated_phrases == []
        assert result.cleaned_text == ""

    def test_empty_recent_messages(self) -> None:
        response = "Всё будет хорошо, я уверен в этом."
        result = check_repetition(response, [])
        assert result.repeated_phrases == []
        assert result.cleaned_text == response

    def test_single_sentence_response_flagged(self) -> None:
        response = "Это нормально хотеть сбежать от рутины."
        recent = ["Это нормально хотеть сбежать от рутины."]
        result = check_repetition(response, recent)
        assert len(result.repeated_phrases) == 1
        assert result.cleaned_text == ""


# ---------------------------------------------------------------------------
# check_repetition — greeting-only edge case
# ---------------------------------------------------------------------------


class TestCheckRepetitionEdgeCases:
    def test_greeting_only_response(self) -> None:
        """A greeting-only response should pass through even if similar greetings were sent."""
        result = check_repetition("Привет!", ["Привет! Как дела?"])
        assert result.cleaned_text == "Привет!"

    def test_all_sentences_repeated_yields_empty(self) -> None:
        response = "Все устают от рутины. Это нормально хотеть сбежать."
        recent = [
            "Все устают от рутины и будничных дел.",
            "Это нормально хотеть сбежать от проблем.",
        ]
        result = check_repetition(response, recent)
        assert len(result.repeated_phrases) >= 1
        # Cleaned text may be empty or minimal

    def test_mixed_languages_no_false_positive(self) -> None:
        response = "I understand your feelings. Давай поговорим об этом."
        recent = ["Мне нравится когда мы разговариваем."]
        result = check_repetition(response, recent)
        assert result.repeated_phrases == []


# ---------------------------------------------------------------------------
# build_anti_repetition_instruction
# ---------------------------------------------------------------------------


class TestAntiRepetitionInstruction:
    def test_single_phrase(self) -> None:
        instruction = build_anti_repetition_instruction(["это нормально хотеть сбежать"])
        assert "это нормально хотеть сбежать" in instruction
        assert "Не повторяй" in instruction

    def test_multiple_phrases_limited_to_3(self) -> None:
        phrases = [f"фраза номер {i}" for i in range(5)]
        instruction = build_anti_repetition_instruction(phrases)
        assert "фраза номер 0" in instruction
        assert "фраза номер 2" in instruction
        assert "фраза номер 4" not in instruction

    def test_empty_phrases(self) -> None:
        instruction = build_anti_repetition_instruction([])
        assert "Не повторяй" in instruction
