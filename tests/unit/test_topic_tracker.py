"""Unit tests for companion_bot_core.orchestrator.topic_tracker."""

from __future__ import annotations

import pytest

from companion_bot_core.orchestrator.topic_tracker import (
    TOPIC_SWITCH_INSTRUCTION,
    TopicSwitchResult,
    detect_topic_switch,
    extract_keywords,
    get_stored_keywords,
    store_topic,
)

# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_basic_extraction(self) -> None:
        kw = extract_keywords("Расскажи про историю Python")
        assert "расскажи" in kw
        assert "историю" in kw
        assert "python" in kw

    def test_filters_stop_words(self) -> None:
        kw = extract_keywords("Я хочу знать что это такое")
        assert "хочу" in kw
        assert "знать" in kw
        # stop words filtered
        assert "что" not in kw
        assert "это" not in kw

    def test_filters_short_tokens(self) -> None:
        kw = extract_keywords("Ну да ок")
        # "ну", "да", "ок" are all <= 2 chars
        assert len(kw) == 0

    def test_empty_string(self) -> None:
        assert extract_keywords("") == frozenset()

    def test_whitespace_only(self) -> None:
        assert extract_keywords("   \n\t  ") == frozenset()

    def test_case_insensitive(self) -> None:
        kw = extract_keywords("PYTHON Programming")
        assert "python" in kw
        assert "programming" in kw

    def test_mixed_languages(self) -> None:
        kw = extract_keywords("Помоги настроить Docker контейнер")
        assert "помоги" in kw
        assert "настроить" in kw
        assert "docker" in kw
        assert "контейнер" in kw


# ---------------------------------------------------------------------------
# Topic switch signal detection
# ---------------------------------------------------------------------------


class TestTopicSwitchSignals:
    @pytest.mark.parametrize(
        "text",
        [
            "Кстати, а что ты думаешь про кино?",
            "А вот ещё хотела спросить",
            "А вот еще интересно",
            "Забей, давай про другое",
            "Ладно, другое — расскажи про книги",
            "Сменим тему, а?",
            "Давай о другом поговорим",
            "Не об этом, расскажи лучше про музыку",
            "Давай про другое",
            "Хватит об этом, надоело",
        ],
    )
    def test_detects_switch_signal(self, text: str) -> None:
        result = detect_topic_switch(text, frozenset({"предыдущая", "тема", "слова"}))
        assert result.switched, f"Expected switch for: {text!r}"
        assert result.signal_score >= 0.35  # noqa: PLR2004

    @pytest.mark.parametrize(
        "text",
        [
            "Расскажи подробнее",
            "Интересно, продолжай",
            "А что дальше?",
            "Понял, спасибо",
        ],
    )
    def test_no_false_positive_signals(self, text: str) -> None:
        # Same topic keywords — should not trigger
        prev = extract_keywords(text)
        result = detect_topic_switch(text, prev)
        assert not result.switched, f"Should not switch for: {text!r}"


# ---------------------------------------------------------------------------
# Keyword divergence detection
# ---------------------------------------------------------------------------


class TestKeywordDivergence:
    def test_different_topics_detected(self) -> None:
        prev = frozenset({"работа", "начальник", "проект", "дедлайн"})
        result = detect_topic_switch(
            "Посоветуй хороший сериал на вечер", prev,
        )
        assert result.switched
        assert result.keyword_overlap < 0.3  # noqa: PLR2004

    def test_same_topic_not_switched(self) -> None:
        prev = frozenset({"сериал", "вечер", "посоветуй"})
        result = detect_topic_switch(
            "А какой сериал на вечер посмотреть?", prev,
        )
        assert not result.switched

    def test_too_few_keywords_no_switch(self) -> None:
        # Single keyword on each side — not enough for meaningful comparison.
        prev = frozenset({"привет"})
        result = detect_topic_switch("Пока", prev)
        assert not result.switched

    def test_empty_previous_no_switch(self) -> None:
        result = detect_topic_switch("Расскажи про космос", frozenset())
        assert not result.switched


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestTopicSwitchResult:
    def test_result_fields(self) -> None:
        result = detect_topic_switch(
            "Кстати, давай про фильмы",
            frozenset({"работа", "начальник"}),
        )
        assert isinstance(result, TopicSwitchResult)
        assert isinstance(result.switched, bool)
        assert isinstance(result.signal_score, float)
        assert isinstance(result.keyword_overlap, float)
        assert isinstance(result.new_keywords, frozenset)

    def test_new_keywords_populated(self) -> None:
        result = detect_topic_switch(
            "Расскажи про машинное обучение",
            frozenset(),
        )
        assert "машинное" in result.new_keywords
        assert "обучение" in result.new_keywords


# ---------------------------------------------------------------------------
# Instruction constant
# ---------------------------------------------------------------------------


class TestInstruction:
    def test_instruction_not_empty(self) -> None:
        assert TOPIC_SWITCH_INSTRUCTION
        assert "тем" in TOPIC_SWITCH_INSTRUCTION.lower()


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


class _FakeRedis:
    """Minimal fake Redis for testing get/set operations."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        self._store[key] = value


class TestRedisHelpers:
    @pytest.mark.asyncio
    async def test_get_stored_keywords_empty(self) -> None:
        redis = _FakeRedis()
        kw = await get_stored_keywords(redis, "user-1")
        assert kw == frozenset()

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self) -> None:
        redis = _FakeRedis()
        keywords = frozenset({"python", "обучение", "книги"})
        await store_topic(redis, "user-1", keywords)
        stored = await get_stored_keywords(redis, "user-1")
        assert stored == keywords

    @pytest.mark.asyncio
    async def test_store_saves_previous(self) -> None:
        redis = _FakeRedis()
        first = frozenset({"работа", "проект"})
        await store_topic(redis, "user-1", first)

        second = frozenset({"кино", "сериал"})
        await store_topic(redis, "user-1", second, save_previous=True)

        # Current should be the new topic.
        current = await get_stored_keywords(redis, "user-1")
        assert current == second

        # Previous should be preserved.
        prev_raw = await redis.get("topic:prev:user-1")
        assert prev_raw is not None
        prev = frozenset(prev_raw.split(",")) if prev_raw else frozenset()
        assert prev == first

    @pytest.mark.asyncio
    async def test_empty_keywords_store(self) -> None:
        redis = _FakeRedis()
        await store_topic(redis, "user-1", frozenset())
        stored = await get_stored_keywords(redis, "user-1")
        assert stored == frozenset()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_message(self) -> None:
        result = detect_topic_switch("", frozenset({"работа", "проект"}))
        assert not result.switched
        assert result.new_keywords == frozenset()

    def test_whitespace_message(self) -> None:
        result = detect_topic_switch("   ", frozenset({"работа", "проект"}))
        assert not result.switched

    def test_case_insensitive_signals(self) -> None:
        result = detect_topic_switch(
            "КСТАТИ, а что думаешь?",
            frozenset({"предыдущая", "тема"}),
        )
        assert result.switched

    def test_mixed_signal_and_keywords(self) -> None:
        # Signal phrase + new keywords — should definitely switch.
        result = detect_topic_switch(
            "Кстати, расскажи про космос и звёзды",
            frozenset({"работа", "начальник", "проект"}),
        )
        assert result.switched
        assert result.signal_score >= 0.35  # noqa: PLR2004
