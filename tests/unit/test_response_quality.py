"""Parametrized quality checks for bot responses — no LLM required.

Loads (input, response) fixtures from ``fixtures/response_pairs.json`` and
validates each pair against deterministic quality checks.  Runs in CI on
every PR with zero network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from companion_bot_core.quality.checks import (
    contains_name,
    count_bullet_points,
    count_sentences,
    has_ai_markers,
    has_menu_pattern,
    is_short_farewell,
    ngram_overlap,
)

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

_FIXTURES_PATH = Path(__file__).parent / "fixtures" / "response_pairs.json"


def _load_pairs() -> list[dict[str, Any]]:
    return json.loads(_FIXTURES_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


_PAIRS = _load_pairs()


def _pair_id(pair: dict[str, Any]) -> str:
    return pair["id"]


# ---------------------------------------------------------------------------
# Parametrized fixture-driven tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", _PAIRS, ids=[_pair_id(p) for p in _PAIRS])
class TestResponseQualityFixtures:
    """Each fixture pair is validated against its declared checks."""

    def test_no_ai_markers(self, pair: dict[str, Any]) -> None:
        if not pair["checks"].get("no_ai_markers"):
            pytest.skip("check not requested")
        markers = has_ai_markers(pair["response"])
        assert markers == [], f"AI markers found: {markers}"

    def test_no_menu_pattern(self, pair: dict[str, Any]) -> None:
        if not pair["checks"].get("no_menu"):
            pytest.skip("check not requested")
        assert not has_menu_pattern(pair["response"]), "Menu pattern detected"

    def test_max_sentences(self, pair: dict[str, Any]) -> None:
        limit = pair["checks"].get("max_sentences")
        if limit is None:
            pytest.skip("check not requested")
        actual = count_sentences(pair["response"])
        assert actual <= limit, f"Too many sentences: {actual} > {limit}"

    def test_is_short_farewell(self, pair: dict[str, Any]) -> None:
        if not pair["checks"].get("is_short_farewell"):
            pytest.skip("check not requested")
        assert is_short_farewell(pair["response"]), "Expected short farewell"

    def test_contains_name(self, pair: dict[str, Any]) -> None:
        name = pair["checks"].get("contains_name")
        if name is None:
            pytest.skip("check not requested")
        assert contains_name(pair["response"], name), f"Name '{name}' not found"


# ---------------------------------------------------------------------------
# Direct unit tests for individual check functions
# ---------------------------------------------------------------------------


class TestHasAiMarkers:
    def test_clean_text(self) -> None:
        assert has_ai_markers("Привет! Как дела?") == []

    def test_detects_russian_ai_marker(self) -> None:
        markers = has_ai_markers("Как языковая модель, я не могу...")
        assert len(markers) >= 1

    def test_detects_english_ai_marker(self) -> None:
        markers = has_ai_markers("As an AI, I don't have feelings")
        assert len(markers) >= 1

    def test_detects_chatbot_marker(self) -> None:
        markers = has_ai_markers("Я всего лишь чат-бот")
        assert len(markers) >= 1

    def test_detects_help_offer_marker(self) -> None:
        markers = has_ai_markers("Чем ещё могу помочь?")
        assert len(markers) >= 1

    def test_detects_specialist_referral(self) -> None:
        markers = has_ai_markers("Обратитесь к специалисту для консультации")
        assert len(markers) >= 1


class TestCountBulletPoints:
    def test_no_bullets(self) -> None:
        assert count_bullet_points("Просто текст без списков.") == 0

    def test_dash_bullets(self) -> None:
        text = "Варианты:\n- Первый\n- Второй\n- Третий"
        assert count_bullet_points(text) == 3  # noqa: PLR2004

    def test_asterisk_bullets(self) -> None:
        text = "* Один\n* Два"
        assert count_bullet_points(text) == 2  # noqa: PLR2004

    def test_mixed_text_with_bullets(self) -> None:
        text = "Заголовок\n- Пункт 1\nТекст\n- Пункт 2"
        assert count_bullet_points(text) == 2  # noqa: PLR2004


class TestCountSentences:
    def test_empty(self) -> None:
        assert count_sentences("") == 0

    def test_single_sentence(self) -> None:
        assert count_sentences("Привет!") == 1

    def test_multiple_sentences(self) -> None:
        assert count_sentences("Первое. Второе! Третье?") == 3  # noqa: PLR2004

    def test_ellipsis(self) -> None:
        assert count_sentences("Ну ладно… Потом поговорим.") == 2  # noqa: PLR2004


class TestHasMenuPattern:
    def test_no_menu(self) -> None:
        assert not has_menu_pattern("Обычный текст без меню.")

    def test_bullet_menu(self) -> None:
        text = "Выбирай:\n- Вариант A\n- Вариант B\n- Вариант C"
        assert has_menu_pattern(text)

    def test_numbered_menu(self) -> None:
        text = "1) Читать\n2) Гулять\n3) Спать"
        assert has_menu_pattern(text)

    def test_numbered_dot_menu(self) -> None:
        text = "1. Работа\n2. Спорт\n3. Хобби"
        assert has_menu_pattern(text)

    def test_two_items_not_menu(self) -> None:
        text = "- Первый\n- Второй"
        assert not has_menu_pattern(text)


class TestIsShortFarewell:
    def test_farewell_short(self) -> None:
        assert is_short_farewell("Пока! Удачи.")

    def test_farewell_goodnight(self) -> None:
        assert is_short_farewell("Спокойной ночи!")

    def test_farewell_long_not_short(self) -> None:
        text = "Пока! Было приятно. Надеюсь ещё увидимся. Желаю удачи во всём."
        assert not is_short_farewell(text)

    def test_not_farewell_at_all(self) -> None:
        assert not is_short_farewell("Расскажи мне про космос.")


class TestContainsName:
    def test_name_found(self) -> None:
        assert contains_name("Привет, Маша! Как дела?", "Маша")

    def test_name_not_found(self) -> None:
        assert not contains_name("Привет! Как дела?", "Маша")

    def test_case_insensitive(self) -> None:
        assert contains_name("привет маша", "Маша")

    def test_empty_name(self) -> None:
        assert not contains_name("Привет!", "")

    def test_partial_match_rejected(self) -> None:
        assert not contains_name("Машина стоит во дворе", "Маша")


class TestNgramOverlap:
    """Tests for the canonical ngram_overlap in quality.checks."""

    def test_identical_texts(self) -> None:
        text = "Это нормально хотеть сбежать от рутины"
        assert ngram_overlap(text, text) == pytest.approx(1.0)

    def test_different_texts(self) -> None:
        a = "Сегодня прекрасная погода за окном"
        b = "Мне нравится читать книги вечером"
        assert ngram_overlap(a, b) == pytest.approx(0.0)

    def test_short_text(self) -> None:
        assert ngram_overlap("да", "нет") == 0.0

    def test_empty(self) -> None:
        assert ngram_overlap("", "текст из нескольких слов") == 0.0

    def test_custom_n(self) -> None:
        text = "раз два три четыре пять"
        assert ngram_overlap(text, text, n=2) == pytest.approx(1.0)
