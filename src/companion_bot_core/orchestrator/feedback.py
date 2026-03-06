"""In-bot user satisfaction feedback collection.

Collects feedback naturally within conversation flow:
- Triggers every N-th session (configurable) at farewell.
- Never asks more than once per ``cooldown_days`` per user.
- Classifies the user's response into a 1-5 sentiment score.
- Saves the result to the ``feedback_entries`` table.

Redis key schema
----------------
``feedback:session_count:{user_uuid}``
    Integer counter of sessions since last feedback ask.  No TTL
    (reset on ask).

``feedback:pending:{user_uuid}``
    Set when the bot has asked for feedback and the next message
    should be classified as a feedback response.  TTL = 5 minutes.

``feedback:last_asked:{user_uuid}``
    Timestamp (ISO) of the last feedback request.  TTL = cooldown
    period (default 7 days).
"""

from __future__ import annotations

import contextlib
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from companion_bot_core.db.models import FeedbackEntry
from companion_bot_core.logging_config import get_logger
from companion_bot_core.privacy.field_encryption import NOOP_ENCRYPTOR, FieldEncryptor

if TYPE_CHECKING:
    from uuid import UUID

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)

_SESSION_COUNT_PREFIX = "feedback:session_count"
_PENDING_PREFIX = "feedback:pending"
_LAST_ASKED_PREFIX = "feedback:last_asked"
_PENDING_TTL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------


async def should_ask_feedback(
    redis: Redis,
    user_id: str,
    *,
    session_interval: int = 10,
    cooldown_days: int = 7,
) -> bool:
    """Return True if the bot should ask for feedback from this user.

    Conditions (all must be met):
    1. The session counter has reached ``session_interval``.
    2. No feedback was asked within the last ``cooldown_days``.
    3. No feedback request is already pending.
    """
    # Check cooldown first (cheapest).
    last_asked = await redis.get(f"{_LAST_ASKED_PREFIX}:{user_id}")
    if last_asked is not None:
        return False

    # Check pending.
    pending = await redis.get(f"{_PENDING_PREFIX}:{user_id}")
    if pending is not None:
        return False

    # Check session counter.
    count_raw = await redis.get(f"{_SESSION_COUNT_PREFIX}:{user_id}")
    count = int(count_raw) if count_raw is not None else 0
    return count >= session_interval


async def increment_session_counter(redis: Redis, user_id: str) -> None:
    """Increment the session counter for *user_id*.

    Called when a session ends with a farewell.
    """
    await redis.incr(f"{_SESSION_COUNT_PREFIX}:{user_id}")


async def try_claim_feedback_ask(
    redis: Redis,
    user_id: str,
    *,
    session_interval: int = 10,
    cooldown_days: int = 7,
) -> bool:
    """Atomically check whether feedback should be asked and claim the slot.

    Combines :func:`should_ask_feedback` with :func:`mark_feedback_asked`
    using ``SET NX`` on the pending key to prevent duplicate prompts when
    two concurrent farewell messages race through the pipeline.

    Returns ``True`` if feedback should be asked (slot was claimed).
    """
    # Cheap checks first (no atomicity needed — false positives just mean
    # we skip asking, which is fine).
    last_asked = await redis.get(f"{_LAST_ASKED_PREFIX}:{user_id}")
    if last_asked is not None:
        return False

    count_raw = await redis.get(f"{_SESSION_COUNT_PREFIX}:{user_id}")
    count = int(count_raw) if count_raw is not None else 0
    if count < session_interval:
        return False

    # Atomic guard: only one concurrent caller wins.
    acquired = await redis.set(
        f"{_PENDING_PREFIX}:{user_id}",
        "1",
        ex=_PENDING_TTL,
        nx=True,
    )
    if not acquired:
        return False

    # Slot claimed — set cooldown and reset counter.
    # If the pipeline fails, release the pending key so the next message
    # is not mistakenly consumed as a feedback response.
    cooldown_seconds = cooldown_days * 86400
    pipe = redis.pipeline(transaction=False)
    pipe.set(
        f"{_LAST_ASKED_PREFIX}:{user_id}",
        datetime.now(tz=UTC).isoformat(),
        ex=cooldown_seconds,
    )
    pipe.delete(f"{_SESSION_COUNT_PREFIX}:{user_id}")
    try:
        await pipe.execute()
    except Exception:
        # With transaction=False, last_asked may have been set before the
        # pipeline error.  Clean up both keys to avoid orphaned cooldown.
        with contextlib.suppress(Exception):
            await redis.delete(
                f"{_PENDING_PREFIX}:{user_id}",
                f"{_LAST_ASKED_PREFIX}:{user_id}",
            )
        raise
    return True


async def mark_feedback_asked(
    redis: Redis,
    user_id: str,
    *,
    cooldown_days: int = 7,
) -> None:
    """Record that feedback was just asked.

    Sets the pending flag (5-min TTL) and the cooldown key.
    Resets the session counter.
    """
    cooldown_seconds = cooldown_days * 86400
    pipe = redis.pipeline(transaction=False)
    pipe.set(f"{_PENDING_PREFIX}:{user_id}", "1", ex=_PENDING_TTL)
    pipe.set(
        f"{_LAST_ASKED_PREFIX}:{user_id}",
        datetime.now(tz=UTC).isoformat(),
        ex=cooldown_seconds,
    )
    pipe.delete(f"{_SESSION_COUNT_PREFIX}:{user_id}")
    await pipe.execute()


async def is_feedback_pending(redis: Redis, user_id: str) -> bool:
    """Return True if the user has a pending feedback request."""
    val = await redis.get(f"{_PENDING_PREFIX}:{user_id}")
    return val is not None


async def clear_feedback_pending(redis: Redis, user_id: str) -> None:
    """Clear the pending feedback flag."""
    await redis.delete(f"{_PENDING_PREFIX}:{user_id}")


async def rollback_feedback_claim(redis: Redis, user_id: str) -> None:
    """Undo the side-effects of :func:`try_claim_feedback_ask`.

    Called when the pipeline fails after a feedback ask was successfully
    claimed but before the reply reached the user.  Removes both the
    ``feedback:pending`` flag (so the next message is not consumed as a
    feedback response) and the ``feedback:last_asked`` cooldown key (so the
    bot can re-ask at the next eligible session instead of silently waiting
    out the cooldown period).

    The session counter is *not* restored because its previous value is
    unknown; the only consequence is that the next feedback trigger may be
    delayed by up to ``session_interval`` sessions.
    """
    pipe = redis.pipeline(transaction=True)
    pipe.delete(f"{_PENDING_PREFIX}:{user_id}")
    pipe.delete(f"{_LAST_ASKED_PREFIX}:{user_id}")
    await pipe.execute()


# ---------------------------------------------------------------------------
# Sentiment classification (regex-based, no LLM call)
# ---------------------------------------------------------------------------

# Explicit numeric scores (1-5) — match at the start of the message
# (followed by punctuation or end) or at the end of the message.  This avoids
# misclassifying "3 слова - ты крутой!" as score 3, while still handling
# "2, но вообще отлично" and "ставлю 5" correctly.
_NUMERIC_START_RE = re.compile(r"^\s*([1-5])(?=\s*(?:$|[,;.!?\-—:]))")
_NUMERIC_END_RE = re.compile(r"\b([1-5])\s*[.!?]*\s*$")

# Positive signal patterns (RU + EN).
_POSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bотлично\b",
        r"\bсупер\b",
        r"\bкласс\b",
        r"\bкруто\b",
        r"\bздорово\b",
        r"\bнрав\w*\b",  # нравится, нравилось
        r"\bхорош\w*\b",  # хорошо, хороший
        r"\bлюблю\b",
        r"\bкайф\b",
        r"\bfire\b",
        r"\bgreat\b",
        r"\bawesome\b",
        r"\blove\b",
        r"\bgood\b",
        r"\bnice\b",
        r"\bexcellent\b",
        r"\bamazing\b",
        r"\bwonderful\b",
    ]
]

# Negative signal patterns (RU + EN).
_NEGATIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bплохо\b",
        r"\bужас\b",
        r"\bотстой\b",
        r"\bскучно\b",
        r"\bне\s*нрав\w*\b",
        r"\bнадоел\w*\b",
        r"\bбесит\b",
        r"\bтупо\b",
        r"\bфигня\b",
        r"\bbad\b",
        r"\bterrible\b",
        r"\bawful\b",
        r"\bhate\b",
        r"\bboring\b",
        r"\bworse?\b",
        r"\bhorrible\b",
    ]
]

# Neutral/moderate patterns.
_NEUTRAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bнорм\w*\b",  # нормально, норм
        r"\bсойдёт\b",
        r"\bок\b",
        r"\bokay\b",
        r"\bfine\b",
        r"\balright\b",
        r"\bso[\-\s]*so\b",
    ]
]


def classify_sentiment(text: str) -> int:
    """Classify user text into a 1-5 sentiment score.

    Strategy:
    1. If text contains an explicit digit 1-5, use it.
    2. Count positive and negative signal matches.
    3. Fall back to neutral (3) for ambiguous input.
    """
    text = text.strip()
    if not text:
        return 3

    # 1. Explicit numeric score (at start or end of message only).
    m = _NUMERIC_START_RE.search(text) or _NUMERIC_END_RE.search(text)
    if m:
        return int(m.group(1))

    # 2. Signal-based scoring.
    pos = sum(1 for p in _POSITIVE_PATTERNS if p.search(text))
    neg = sum(1 for p in _NEGATIVE_PATTERNS if p.search(text))
    neu = sum(1 for p in _NEUTRAL_PATTERNS if p.search(text))

    if pos > 0 and neg == 0:
        return 5 if pos >= 2 else 4
    if neg > 0 and pos == 0:
        return 1 if neg >= 2 else 2
    if pos > 0 and neg > 0:
        return 3
    if neu > 0:
        return 3

    # 3. Default: neutral.
    return 3


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def save_feedback(
    db_session: AsyncSession,
    user_id: UUID,
    raw_text: str,
    sentiment_score: int,
    *,
    session_id: UUID | None = None,
    encryptor: FieldEncryptor | None = None,
) -> FeedbackEntry:
    """Persist a feedback entry to the database."""
    enc = encryptor or NOOP_ENCRYPTOR
    entry = FeedbackEntry(
        user_id=user_id,
        session_id=session_id,
        raw_text=enc.encrypt(raw_text),
        sentiment_score=sentiment_score,
    )
    db_session.add(entry)
    await db_session.flush()
    log.info(
        "feedback_saved",
        user_id=str(user_id),
        sentiment_score=sentiment_score,
    )
    return entry
