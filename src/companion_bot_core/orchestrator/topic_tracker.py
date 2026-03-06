"""Conversation topic tracker: detect topic switches and inject instructions.

Tracks the current conversation topic per user in Redis.  When the user
switches topics (detected via signal phrases or keyword divergence), an
instruction is injected into the system prompt telling the model not to
return to the previous topic.

Public surface:
    TopicSwitchResult  — result of topic switch detection
    extract_keywords   — extract content keywords from a message
    detect_topic_switch — detect whether a topic switch occurred
    TOPIC_SWITCH_INSTRUCTION — instruction injected on switch
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, NamedTuple

from companion_bot_core.signals import Signal, compile_signals, score_signals

if TYPE_CHECKING:
    from redis.asyncio import Redis

# ---------------------------------------------------------------------------
# Topic switch signal phrases
# ---------------------------------------------------------------------------

_SWITCH_SIGNALS: Final[list[Signal]] = compile_signals([
    (r"\bкстати\b", 0.5),
    (r"\bа вот ещё\b", 0.5),
    (r"\bа вот еще\b", 0.5),
    (r"\bзабей\b", 0.5),
    (r"\bладно,?\s*друг(ое|ая тема)\b", 0.6),
    (r"\bсменим тему\b", 0.7),
    (r"\bдавай о другом\b", 0.6),
    (r"\bне об этом\b", 0.5),
    (r"\bдавай про другое\b", 0.6),
    (r"\bхватит об этом\b", 0.6),
    (r"\bну ладно\b.{0,15}\b(а |давай|расскажи|что)\b", 0.5),
    (r"\bвообще,?\s*я\b", 0.4),
    (r"\bа (?:ещё|еще)\b", 0.4),
    (r"\bслушай,?\s*а\b", 0.4),
])

_SWITCH_THRESHOLD: Final[float] = 0.35

# Instruction injected into the system prompt on topic switch.
TOPIC_SWITCH_INSTRUCTION: Final[str] = (
    "Пользователь перешёл к новой теме. "
    "Не возвращайся к предыдущей теме, отвечай по новой."
)

# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

# Common Russian stop words to filter out.
_STOP_WORDS: Final[frozenset[str]] = frozenset({
    "а", "без", "более", "бы", "был", "была", "были", "было", "быть",
    "в", "вам", "вас", "весь", "во", "вот", "всё", "все", "всего", "всех",
    "вы", "где", "да", "даже", "для", "до", "его", "ее", "её", "если",
    "есть", "ещё", "еще", "ж", "же", "за", "и", "из", "или", "им", "их",
    "к", "как", "ко", "когда", "кто", "ли", "либо", "мне", "может",
    "мой", "моя", "моё", "мы", "на", "надо", "наш", "не", "нет", "ни",
    "них", "но", "ну", "о", "об", "он", "она", "они", "оно", "от",
    "очень", "по", "под", "при", "про", "с", "свой", "себя", "сейчас",
    "со", "та", "так", "такой", "там", "те", "тебе", "тебя", "тем",
    "то", "тоже", "того", "только", "тот", "ту", "ты", "у", "уже",
    "хотя", "чего", "чем", "что", "чтобы", "чья", "эта", "эти",
    "это", "этого", "этой", "этом", "этот", "я",
})

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Minimum keyword overlap ratio to consider topics "same".
_KEYWORD_OVERLAP_THRESHOLD: Final[float] = 0.3

# Minimum number of keywords required for overlap comparison.
_MIN_KEYWORDS: Final[int] = 2


def extract_keywords(text: str) -> frozenset[str]:
    """Extract content keywords from *text*, filtering stop words.

    Returns a frozenset of lowercase keyword strings.
    """
    if not text or not text.strip():
        return frozenset()

    tokens = _WORD_RE.findall(text.lower())
    return frozenset(
        t for t in tokens
        if t not in _STOP_WORDS and len(t) > 2  # noqa: PLR2004
    )


def _keyword_overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute Jaccard similarity between two keyword sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union


# ---------------------------------------------------------------------------
# Topic switch detection
# ---------------------------------------------------------------------------


class TopicSwitchResult(NamedTuple):
    """Result of topic switch detection."""

    switched: bool
    signal_score: float
    keyword_overlap: float
    new_keywords: frozenset[str]


def detect_topic_switch(
    text: str,
    previous_keywords: frozenset[str],
) -> TopicSwitchResult:
    """Detect whether *text* represents a topic switch.

    A switch is detected when:
    1. Explicit switch signal phrases score above threshold, OR
    2. Keyword overlap with *previous_keywords* is below threshold
       (and both sets have enough keywords for meaningful comparison).

    Args:
        text:               The raw user message.
        previous_keywords:  Keywords from the previous topic (from Redis).

    Returns:
        A :class:`TopicSwitchResult` with detection details.
    """
    new_keywords = extract_keywords(text)
    signal_score = score_signals(text, _SWITCH_SIGNALS)

    # Explicit switch signal detected.
    if signal_score >= _SWITCH_THRESHOLD:
        return TopicSwitchResult(
            switched=True,
            signal_score=signal_score,
            keyword_overlap=_keyword_overlap(new_keywords, previous_keywords),
            new_keywords=new_keywords,
        )

    # Keyword divergence: only trigger if both sides have enough keywords
    # for a meaningful comparison.
    overlap = _keyword_overlap(new_keywords, previous_keywords)
    if (
        len(new_keywords) >= _MIN_KEYWORDS
        and len(previous_keywords) >= _MIN_KEYWORDS
        and overlap < _KEYWORD_OVERLAP_THRESHOLD
    ):
        return TopicSwitchResult(
            switched=True,
            signal_score=signal_score,
            keyword_overlap=overlap,
            new_keywords=new_keywords,
        )

    return TopicSwitchResult(
        switched=False,
        signal_score=signal_score,
        keyword_overlap=overlap,
        new_keywords=new_keywords,
    )


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

# Redis key layout:
#   topic:{user_uuid}        — JSON-encoded current topic keywords (30-min TTL)
#   topic:prev:{user_uuid}   — JSON-encoded previous topic keywords (30-min TTL)

_TOPIC_KEY_PREFIX: Final[str] = "topic"
_TOPIC_PREV_PREFIX: Final[str] = "topic:prev"
_TOPIC_TTL: Final[int] = 1800  # 30 minutes


def _topic_key(user_id: str) -> str:
    return f"{_TOPIC_KEY_PREFIX}:{user_id}"


def _topic_prev_key(user_id: str) -> str:
    return f"{_TOPIC_PREV_PREFIX}:{user_id}"


async def get_stored_keywords(redis: Redis, user_id: str) -> frozenset[str]:
    """Load the stored topic keywords from Redis.

    Returns an empty frozenset if no topic is stored.
    """
    key = _topic_key(user_id)
    raw: bytes | str | None = await redis.get(key)
    if raw is None:
        return frozenset()
    text = raw.decode() if isinstance(raw, bytes) else raw
    return frozenset(text.split(",")) if text else frozenset()


async def store_topic(
    redis: Redis,
    user_id: str,
    keywords: frozenset[str],
    *,
    save_previous: bool = False,
) -> None:
    """Store topic keywords in Redis with TTL.

    When *save_previous* is ``True``, the current topic is moved to the
    ``topic:prev`` key before overwriting (for warm-return feature).
    """
    key = _topic_key(user_id)

    if save_previous:
        current_raw: bytes | str | None = await redis.get(key)
        if current_raw is not None:
            prev_key = _topic_prev_key(user_id)
            await redis.set(prev_key, current_raw, ex=_TOPIC_TTL)

    value = ",".join(sorted(keywords)) if keywords else ""
    await redis.set(key, value, ex=_TOPIC_TTL)
