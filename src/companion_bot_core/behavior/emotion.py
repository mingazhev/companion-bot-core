"""Emotion mode classifier for user messages.

Classifies user message emotion into one of five modes using regex-based
signal scoring (same approach as behavior detection in ``detector.py``).
The detected mode drives a mode-specific instruction that is injected into
the system prompt before inference.

Public surface:
    EmotionMode        — literal type for the five emotion modes
    EmotionResult      — classifier output (mode + confidence)
    detect_emotion     — classify a user message into an emotion mode
    EMOTION_INSTRUCTIONS — mapping of mode → instruction to inject
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, Field

from companion_bot_core.signals import Signal, compile_signals, score_signals

# Five possible emotion modes.
EmotionMode = Literal["venting", "validation", "task", "farewell", "neutral"]

# ---------------------------------------------------------------------------
# Signal patterns per emotion mode
# ---------------------------------------------------------------------------

_VENTING_SIGNALS = compile_signals([
    (r"\bустал[аи]?\b", 0.5),
    (r"\bбесит\b", 0.6),
    (r"\bдостал[аои]?\b", 0.5),
    (r"\bтяжело\b", 0.4),
    (r"\bобидно\b", 0.5),
    (r"\bрасстроен[а]?\b", 0.5),
    (r"\bгрустн[оа]\b", 0.4),
    (r"\bзл[оюа]\b", 0.4),
    (r"\bненавижу\b", 0.6),
    (r"\bзадолбал[аои]?\b", 0.6),
    (r"\bсил нет\b", 0.5),
    (r"\bне могу больше\b", 0.6),
    (r"\bвыгорел[а]?\b", 0.5),
    (r"\bплохо\b", 0.3),
    (r"\bтошнит\b", 0.4),
    (r"\bзаколебал[аои]?\b", 0.5),
    (r"\bопять\b.{0,20}\b(не|нет|провал|отказ)\b", 0.4),
    (r"\bпечальн[оа]\b", 0.4),
    (r"\bвымотал[аои]?\b", 0.5),
    (r"\bстрадаю\b", 0.5),
])

_VALIDATION_SIGNALS = compile_signals([
    (r"\bнормально\s+ли\b", 0.6),
    (r"\bэто\s+(ок|нормально|норм)\b", 0.5),
    (r"\bя\s+не\s+туп(ой|ая)\b", 0.7),
    (r"\bтак\s+бывает\b", 0.5),
    (r"\bя\s+(не\s+)?прав[а]?\b", 0.4),
    (r"\bсо\s+мной\s+(всё|все)\s+(ок|нормально|норм)\b", 0.6),
    (r"\bэто\s+не\s+страшно\b", 0.5),
    (r"\bтак\s+можно\b", 0.5),
    (r"\bя\s+(не\s+)?сумасшедш(ий|ая)\b", 0.6),
    (r"\bмне\s+можно\b", 0.4),
    (r"\bя\s+не\s+(лентяй(ка)?|неудачни[кц]а?)\b", 0.5),
])

_FAREWELL_SIGNALS = compile_signals([
    (r"^пока[!.\s]*$", 0.7),
    (r"\bну\s+пока\b", 0.6),
    (r"\bладно,?\s+пока\b", 0.6),
    (r"\bдо\s+завтра\b", 0.7),
    (r"\bспокойной\s+ночи\b", 0.7),
    (r"\bспасибо,?\s*(всё|все)\b", 0.6),
    (r"\bдо\s+связи\b", 0.6),
    (r"\bдо\s+встречи\b", 0.6),
    (r"\bпойду\b", 0.3),
    (r"\bвсё,?\s+пока\b", 0.7),
    (r"\bспасибо,?\s+пока\b", 0.7),
    (r"\bна\s+сегодня\s+(всё|все)\b", 0.6),
    (r"\bдобрых\s+снов\b", 0.6),
])

_TASK_SIGNALS = compile_signals([
    (r"\bчто\s+такое\b", 0.5),
    (r"\bкак\s+сделать\b", 0.5),
    (r"\bобъясни\b", 0.5),
    (r"\bпосоветуй\b", 0.5),
    (r"\bрасскажи\s+про\b", 0.4),
    (r"\bпомоги\s+(с|мне)\b", 0.4),
    (r"\bкак\s+(работает|настроить|установить|подключить)\b", 0.5),
    (r"\bнайди\b", 0.4),
    (r"\bпорекомендуй\b", 0.5),
    (r"\bподскажи\b", 0.4),
    (r"\bсравни\b", 0.4),
    (r"\bнапиши\b", 0.4),
    (r"\bсоставь\b", 0.4),
])

# Ordered by priority: venting/validation checked before task/farewell.
_MODE_SIGNALS: Final[dict[EmotionMode, list[Signal]]] = {
    "venting": _VENTING_SIGNALS,
    "validation": _VALIDATION_SIGNALS,
    "farewell": _FAREWELL_SIGNALS,
    "task": _TASK_SIGNALS,
}

# Minimum score to consider a mode detected (below this → neutral).
_EMOTION_THRESHOLD: Final[float] = 0.35

# ---------------------------------------------------------------------------
# Mode → instruction mapping
# ---------------------------------------------------------------------------

EMOTION_INSTRUCTIONS: Final[dict[EmotionMode, str]] = {
    "venting": (
        "Пользователь делится эмоциями. Только эмпатия. "
        "НЕ предлагай решения, планы, варианты."
    ),
    "validation": (
        "Это вопрос-валидация. Ответь коротко и тепло, "
        "2-3 предложения макс."
    ),
    "farewell": (
        "Пользователь прощается. Кратко и тепло. "
        "Не повторяй информацию из разговора."
    ),
    "task": "",
    "neutral": "",
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class EmotionResult(BaseModel):
    """Output of the emotion mode classifier."""

    mode: EmotionMode = Field(description="Detected emotion mode")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score in [0.0, 1.0]",
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def detect_emotion(text: str) -> EmotionResult:
    """Classify *text* into an :class:`EmotionMode`.

    Scores the text against each mode's signal set and picks the highest
    scorer above :data:`_EMOTION_THRESHOLD`.  Falls back to ``neutral``
    when no mode scores high enough.

    Args:
        text: The raw user message.

    Returns:
        An :class:`EmotionResult` with the detected mode and confidence.
    """
    if not text or not text.strip():
        return EmotionResult(mode="neutral", confidence=0.0)

    scores: dict[EmotionMode, float] = {
        mode: score_signals(text, signals)
        for mode, signals in _MODE_SIGNALS.items()
    }

    best_mode: EmotionMode = max(scores, key=lambda m: scores[m])
    best_score = scores[best_mode]

    if best_score < _EMOTION_THRESHOLD:
        return EmotionResult(mode="neutral", confidence=0.0)

    return EmotionResult(mode=best_mode, confidence=best_score)
