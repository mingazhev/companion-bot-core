"""Unit tests for in-bot feedback collection (3.3).

Tests cover:
- feedback.classify_sentiment: numeric, positive, negative, neutral, edge cases
- feedback.should_ask_feedback: session counter, cooldown, pending check
- feedback.increment_session_counter / mark_feedback_asked / clear_feedback_pending
- feedback.save_feedback: DB persistence
- Orchestrator integration: feedback pending processing, feedback trigger on farewell
"""

from __future__ import annotations

import uuid
from typing import Any

import fakeredis.aioredis as fakeredis
import pytest

from companion_bot_core.db.models import FeedbackEntry
from companion_bot_core.orchestrator.feedback import (
    _LAST_ASKED_PREFIX,
    _PENDING_PREFIX,
    _SESSION_COUNT_PREFIX,
    classify_sentiment,
    clear_feedback_pending,
    increment_session_counter,
    is_feedback_pending,
    mark_feedback_asked,
    save_feedback,
    should_ask_feedback,
    try_claim_feedback_ask,
)

# ---------------------------------------------------------------------------
# classify_sentiment
# ---------------------------------------------------------------------------


class TestClassifySentiment:
    def test_explicit_score_1(self) -> None:
        assert classify_sentiment("1") == 1

    def test_explicit_score_5(self) -> None:
        assert classify_sentiment("5") == 5

    def test_explicit_score_3_in_text(self) -> None:
        assert classify_sentiment("я бы поставил 3") == 3

    def test_positive_single_word_ru(self) -> None:
        assert classify_sentiment("отлично") >= 4

    def test_positive_single_word_en(self) -> None:
        assert classify_sentiment("great") >= 4

    def test_strong_positive_ru(self) -> None:
        assert classify_sentiment("супер, отлично!") == 5

    def test_strong_positive_en(self) -> None:
        assert classify_sentiment("amazing, love it!") == 5

    def test_negative_single_word_ru(self) -> None:
        assert classify_sentiment("плохо") <= 2

    def test_negative_single_word_en(self) -> None:
        assert classify_sentiment("terrible") <= 2

    def test_strong_negative_ru(self) -> None:
        assert classify_sentiment("ужас, отстой") == 1

    def test_strong_negative_en(self) -> None:
        assert classify_sentiment("awful, horrible") == 1

    def test_neutral_ru(self) -> None:
        assert classify_sentiment("нормально") == 3

    def test_neutral_en(self) -> None:
        assert classify_sentiment("fine") == 3

    def test_mixed_signals(self) -> None:
        assert classify_sentiment("хорошо, но иногда плохо") == 3

    def test_empty_string(self) -> None:
        assert classify_sentiment("") == 3

    def test_whitespace_only(self) -> None:
        assert classify_sentiment("   ") == 3

    def test_unrelated_text(self) -> None:
        # No signals detected -> neutral
        assert classify_sentiment("просто текст без оценки") == 3

    def test_case_insensitive(self) -> None:
        assert classify_sentiment("ОТЛИЧНО") >= 4
        assert classify_sentiment("TERRIBLE") <= 2

    def test_explicit_score_overrides_words(self) -> None:
        # "2" is found first even though "отлично" is positive
        assert classify_sentiment("2, но вообще отлично") == 2

    def test_ok_is_neutral(self) -> None:
        assert classify_sentiment("ок") == 3


# ---------------------------------------------------------------------------
# Redis-backed trigger logic
# ---------------------------------------------------------------------------


@pytest.fixture()
async def redis() -> fakeredis.FakeRedis:
    client: fakeredis.FakeRedis = fakeredis.FakeRedis(decode_responses=True)
    yield client  # type: ignore[misc]
    await client.aclose()


class TestShouldAskFeedback:
    @pytest.mark.asyncio
    async def test_returns_false_when_counter_below_threshold(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await redis.set(f"{_SESSION_COUNT_PREFIX}:{uid}", "5")
        result = await should_ask_feedback(
            redis, uid, session_interval=10,  # type: ignore[arg-type]
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_counter_at_threshold(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await redis.set(f"{_SESSION_COUNT_PREFIX}:{uid}", "10")
        result = await should_ask_feedback(
            redis, uid, session_interval=10,  # type: ignore[arg-type]
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_counter_above_threshold(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await redis.set(f"{_SESSION_COUNT_PREFIX}:{uid}", "15")
        result = await should_ask_feedback(
            redis, uid, session_interval=10,  # type: ignore[arg-type]
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_cooldown_active(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await redis.set(f"{_SESSION_COUNT_PREFIX}:{uid}", "20")
        await redis.set(f"{_LAST_ASKED_PREFIX}:{uid}", "2026-01-01T00:00:00Z")
        result = await should_ask_feedback(
            redis, uid, session_interval=10,  # type: ignore[arg-type]
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_pending(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await redis.set(f"{_SESSION_COUNT_PREFIX}:{uid}", "20")
        await redis.set(f"{_PENDING_PREFIX}:{uid}", "1")
        result = await should_ask_feedback(
            redis, uid, session_interval=10,  # type: ignore[arg-type]
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_counter(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        result = await should_ask_feedback(
            redis, uid, session_interval=10,  # type: ignore[arg-type]
        )
        assert result is False


class TestIncrementSessionCounter:
    @pytest.mark.asyncio
    async def test_increments_from_zero(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await increment_session_counter(redis, uid)  # type: ignore[arg-type]
        val = await redis.get(f"{_SESSION_COUNT_PREFIX}:{uid}")
        assert val == "1"

    @pytest.mark.asyncio
    async def test_increments_existing(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await redis.set(f"{_SESSION_COUNT_PREFIX}:{uid}", "5")
        await increment_session_counter(redis, uid)  # type: ignore[arg-type]
        val = await redis.get(f"{_SESSION_COUNT_PREFIX}:{uid}")
        assert val == "6"


class TestMarkFeedbackAsked:
    @pytest.mark.asyncio
    async def test_sets_pending_and_cooldown_and_resets_counter(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await redis.set(f"{_SESSION_COUNT_PREFIX}:{uid}", "10")

        await mark_feedback_asked(
            redis, uid, cooldown_days=7,  # type: ignore[arg-type]
        )

        # Pending should be set.
        assert await redis.get(f"{_PENDING_PREFIX}:{uid}") == "1"
        # Cooldown should be set.
        assert await redis.get(f"{_LAST_ASKED_PREFIX}:{uid}") is not None
        # Counter should be reset.
        assert await redis.get(f"{_SESSION_COUNT_PREFIX}:{uid}") is None


class TestIsFeedbackPending:
    @pytest.mark.asyncio
    async def test_false_when_not_set(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        assert await is_feedback_pending(redis, uid) is False  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_true_when_set(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await redis.set(f"{_PENDING_PREFIX}:{uid}", "1")
        assert await is_feedback_pending(redis, uid) is True  # type: ignore[arg-type]


class TestTryClaimFeedbackAsk:
    @pytest.mark.asyncio
    async def test_cleans_up_last_asked_on_pipeline_error(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        """When the internal Redis pipeline fails after partially executing,
        both pending and last_asked keys must be cleaned up."""
        uid = str(uuid.uuid4())
        # Set counter above threshold so claim proceeds.
        await redis.set(f"{_SESSION_COUNT_PREFIX}:{uid}", "10")

        # Patch pipeline.execute to simulate partial failure:
        # The SET for last_asked may have executed before the error.
        original_pipeline = redis.pipeline

        async def failing_pipeline_execute(self_pipe: object) -> None:
            # Simulate: last_asked was set before the error
            await redis.set(f"{_LAST_ASKED_PREFIX}:{uid}", "simulated")
            msg = "connection lost"
            raise ConnectionError(msg)

        def patched_pipeline(**kwargs: object) -> object:
            pipe = original_pipeline(**kwargs)
            pipe.execute = lambda: failing_pipeline_execute(pipe)
            return pipe

        redis.pipeline = patched_pipeline  # type: ignore[assignment]

        with pytest.raises(ConnectionError):
            await try_claim_feedback_ask(redis, uid, session_interval=10)

        # Both keys should be cleaned up.
        assert await redis.get(f"{_PENDING_PREFIX}:{uid}") is None
        assert await redis.get(f"{_LAST_ASKED_PREFIX}:{uid}") is None


class TestClearFeedbackPending:
    @pytest.mark.asyncio
    async def test_clears_pending(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        uid = str(uuid.uuid4())
        await redis.set(f"{_PENDING_PREFIX}:{uid}", "1")
        await clear_feedback_pending(redis, uid)  # type: ignore[arg-type]
        assert await redis.get(f"{_PENDING_PREFIX}:{uid}") is None


# ---------------------------------------------------------------------------
# save_feedback (DB persistence)
# ---------------------------------------------------------------------------


class FakeDbSession:
    """Minimal async session stub for save_feedback tests."""

    def __init__(self) -> None:
        self._store: list[Any] = []

    def add(self, obj: Any) -> None:
        self._store.append(obj)

    async def flush(self) -> None:
        pass


class TestSaveFeedback:
    @pytest.mark.asyncio
    async def test_saves_entry(self) -> None:
        db = FakeDbSession()
        user_id = uuid.uuid4()

        entry = await save_feedback(
            db, user_id, "отлично", 5,  # type: ignore[arg-type]
        )

        assert len(db._store) == 1
        assert isinstance(entry, FeedbackEntry)
        assert entry.user_id == user_id
        assert entry.raw_text == "отлично"
        assert entry.sentiment_score == 5
        assert entry.session_id is None

    @pytest.mark.asyncio
    async def test_saves_with_session_id(self) -> None:
        db = FakeDbSession()
        user_id = uuid.uuid4()
        session_id = uuid.uuid4()

        entry = await save_feedback(
            db,  # type: ignore[arg-type]
            user_id,
            "хорошо",
            4,
            session_id=session_id,
        )

        assert entry.session_id == session_id


# ---------------------------------------------------------------------------
# Metric objects
# ---------------------------------------------------------------------------


class TestFeedbackMetrics:
    def test_user_feedback_score_accepts_observation(self) -> None:
        from companion_bot_core.metrics import USER_FEEDBACK_SCORE

        USER_FEEDBACK_SCORE.observe(4)

    def test_feedback_asked_increments(self) -> None:
        from companion_bot_core.metrics import FEEDBACK_ASKED

        FEEDBACK_ASKED.inc()
