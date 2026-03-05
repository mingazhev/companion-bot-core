"""Unit tests for companion_bot_core.behavior.emotion — emotion mode classifier."""

from __future__ import annotations

import pytest

from companion_bot_core.behavior.emotion import (
    EMOTION_INSTRUCTIONS,
    EmotionResult,
    detect_emotion,
)

# ---------------------------------------------------------------------------
# Venting mode
# ---------------------------------------------------------------------------


class TestVentingDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Я так устала от этого всего",
            "Меня бесит моя работа",
            "Достало уже, не могу больше",
            "Тяжело сейчас, сил нет",
            "Мне обидно что так вышло",
            "Я расстроена из-за ссоры",
            "Мне так грустно сегодня",
            "Ненавижу свою жизнь иногда",
            "Задолбали все эти проблемы",
            "Выгорела на работе полностью",
            "Вымотала эта неделя",
            "Страдаю от бессонницы",
        ],
    )
    def test_detects_venting(self, text: str) -> None:
        result = detect_emotion(text)
        assert result.mode == "venting", f"Expected venting for: {text!r}, got {result.mode}"
        assert result.confidence >= 0.35  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Validation mode
# ---------------------------------------------------------------------------


class TestValidationDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Это нормально ли так чувствовать?",
            "Это ок что я так поступил?",
            "Я не тупой что не понимаю это?",
            "Так бывает наверное у всех?",
            "Со мной всё нормально?",
            "Это не страшно что я провалил?",
            "Так можно делать или нет?",
            "Я не сумасшедший?",
            "Я не лентяй что отдыхаю?",
            "Я не неудачник?",
        ],
    )
    def test_detects_validation(self, text: str) -> None:
        result = detect_emotion(text)
        assert result.mode == "validation", f"Expected validation for: {text!r}, got {result.mode}"
        assert result.confidence >= 0.35  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Farewell mode
# ---------------------------------------------------------------------------


class TestFarewellDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Ладно, пока!",
            "До завтра, спасибо!",
            "Спокойной ночи",
            "Спасибо, всё понятно, пока",
            "До связи!",
            "Всё, пока!",
            "Спасибо, пока!",
            "На сегодня всё",
            "Добрых снов!",
            "До встречи!",
        ],
    )
    def test_detects_farewell(self, text: str) -> None:
        result = detect_emotion(text)
        assert result.mode == "farewell", f"Expected farewell for: {text!r}, got {result.mode}"
        assert result.confidence >= 0.35  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Task mode
# ---------------------------------------------------------------------------


class TestTaskDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Что такое машинное обучение?",
            "Как сделать пиццу дома?",
            "Объясни мне теорему Пифагора",
            "Посоветуй хороший фильм",
            "Расскажи про историю Python",
            "Помоги мне с домашним заданием",
            "Как работает блокчейн?",
            "Найди мне рецепт борща",
            "Порекомендуй книгу про психологию",
            "Подскажи как настроить роутер",
            "Напиши мне план тренировки",
        ],
    )
    def test_detects_task(self, text: str) -> None:
        result = detect_emotion(text)
        assert result.mode == "task", f"Expected task for: {text!r}, got {result.mode}"
        assert result.confidence >= 0.35  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Neutral (default fallback)
# ---------------------------------------------------------------------------


class TestNeutralDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Привет",
            "Как дела?",
            "Ну ок",
            "Хм интересно",
            "Сегодня хорошая погода",
            "Я гулял по парку",
            "",
            "   ",
        ],
    )
    def test_detects_neutral(self, text: str) -> None:
        result = detect_emotion(text)
        assert result.mode == "neutral", f"Expected neutral for: {text!r}, got {result.mode}"
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Instruction mapping
# ---------------------------------------------------------------------------


class TestEmotionInstructions:
    def test_venting_has_instruction(self) -> None:
        assert EMOTION_INSTRUCTIONS["venting"]
        assert "эмпатия" in EMOTION_INSTRUCTIONS["venting"].lower()

    def test_validation_has_instruction(self) -> None:
        assert EMOTION_INSTRUCTIONS["validation"]
        assert "коротко" in EMOTION_INSTRUCTIONS["validation"].lower()

    def test_farewell_has_instruction(self) -> None:
        assert EMOTION_INSTRUCTIONS["farewell"]
        assert "кратко" in EMOTION_INSTRUCTIONS["farewell"].lower()

    def test_task_has_empty_instruction(self) -> None:
        assert EMOTION_INSTRUCTIONS["task"] == ""

    def test_neutral_has_empty_instruction(self) -> None:
        assert EMOTION_INSTRUCTIONS["neutral"] == ""


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class TestEmotionResult:
    def test_schema_fields(self) -> None:
        result = EmotionResult(mode="venting", confidence=0.8)
        assert result.mode == "venting"
        assert result.confidence == 0.8  # noqa: PLR2004

    def test_confidence_clamped_at_one(self) -> None:
        result = EmotionResult(mode="neutral", confidence=1.0)
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self) -> None:
        result = detect_emotion("")
        assert result.mode == "neutral"
        assert result.confidence == 0.0

    def test_whitespace_only(self) -> None:
        result = detect_emotion("   \n\t  ")
        assert result.mode == "neutral"
        assert result.confidence == 0.0

    def test_case_insensitive(self) -> None:
        result = detect_emotion("УСТАЛА ОТ ВСЕГО")
        assert result.mode == "venting"

    def test_mixed_signals_picks_strongest(self) -> None:
        # "бесит" (venting 0.6) should win over weak task signals
        result = detect_emotion("Бесит что не могу разобраться")
        assert result.mode == "venting"
