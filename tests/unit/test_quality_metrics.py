"""Unit tests for conversation quality metrics (3.1).

Tests cover:
- _track_session_message helper: session counting, boundary detection, farewell
- Quality metric emissions in process_message: response_length_sentences,
  farewell_detected_total, session_messages_total
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import fakeredis.aioredis as fakeredis
import pytest

from companion_bot_core.behavior.schemas import DetectionResult
from companion_bot_core.inference.schemas import (
    InferenceReply,
    OpenAIResponse,
    SafetyFlags,
    TokenUsage,
)
from companion_bot_core.orchestrator.orchestrator import (
    _SESSION_COUNT_PREFIX,
    _SESSION_PREV_COUNT_PREFIX,
    _track_session_message,
    process_message,
)
from companion_bot_core.prompt.snapshot_store import InMemorySnapshotStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inference_reply(text: str = "Hello!", total_tokens: int = 42) -> InferenceReply:
    return InferenceReply(
        reply=text,
        usage=TokenUsage(prompt_tokens=30, completion_tokens=12, total_tokens=total_tokens),
        safety_flags=SafetyFlags(
            content_filtered=False, refusal=False, finish_reason="stop",
        ),
    )


def _make_openai_response(content: str = "Hello!") -> OpenAIResponse:
    raw = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
    }
    return OpenAIResponse.model_validate(raw)


def _make_mock_client(reply: InferenceReply) -> MagicMock:
    client = AsyncMock()
    client.chat_completion = AsyncMock(return_value=_make_openai_response(reply.reply))
    return client


def _make_session() -> AsyncMock:
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.begin_nested = MagicMock(return_value=AsyncMock())
    return session


def _make_detection(
    intent: str = "normal_chat",
    risk_level: str = "low",
    confidence: float = 0.9,
    action: str = "pass_through",
) -> DetectionResult:
    return DetectionResult(
        intent=intent,  # type: ignore[arg-type]
        risk_level=risk_level,  # type: ignore[arg-type]
        confidence=confidence,
        action=action,  # type: ignore[arg-type]
        clarification_question=None,
    )


# ---------------------------------------------------------------------------
# _track_session_message
# ---------------------------------------------------------------------------


class TestTrackSessionMessage:
    @pytest.mark.asyncio
    async def test_first_message_creates_session_counter(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=True)
        uid = "user-1"

        await _track_session_message(redis, uid)

        key = f"{_SESSION_COUNT_PREFIX}:{uid}"
        assert await redis.get(key) == "1"

    @pytest.mark.asyncio
    async def test_increments_on_subsequent_messages(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=True)
        uid = "user-2"

        await _track_session_message(redis, uid)
        await _track_session_message(redis, uid)
        await _track_session_message(redis, uid)

        key = f"{_SESSION_COUNT_PREFIX}:{uid}"
        assert await redis.get(key) == "3"

    @pytest.mark.asyncio
    async def test_updates_prev_count_on_each_message(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=True)
        uid = "user-3"

        await _track_session_message(redis, uid)
        await _track_session_message(redis, uid)

        prev_key = f"{_SESSION_PREV_COUNT_PREFIX}:{uid}"
        assert await redis.get(prev_key) == "2"

    @pytest.mark.asyncio
    async def test_farewell_observes_session_length(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=True)
        uid = "user-4"

        await _track_session_message(redis, uid)
        await _track_session_message(redis, uid)
        await _track_session_message(redis, uid)

        with patch(
            "companion_bot_core.orchestrator.orchestrator.SESSION_MESSAGES"
        ) as mock_hist:
            await _track_session_message(redis, uid, is_farewell=True)
            mock_hist.observe.assert_called_once_with(4)

    @pytest.mark.asyncio
    async def test_new_session_observes_previous_session_length(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=True)
        uid = "user-5"

        # First session: 3 messages
        await _track_session_message(redis, uid)
        await _track_session_message(redis, uid)
        await _track_session_message(redis, uid)

        # Simulate session expiry by deleting the counter (TTL expired)
        session_key = f"{_SESSION_COUNT_PREFIX}:{uid}"
        await redis.delete(session_key)

        # New session starts — should observe prev count (3)
        with patch(
            "companion_bot_core.orchestrator.orchestrator.SESSION_MESSAGES"
        ) as mock_hist:
            await _track_session_message(redis, uid)
            mock_hist.observe.assert_called_once_with(3)

    @pytest.mark.asyncio
    async def test_redis_error_does_not_raise(self) -> None:
        redis = AsyncMock()
        redis.incr = AsyncMock(side_effect=ConnectionError("Redis down"))
        uid = "user-6"

        # Should not raise
        await _track_session_message(redis, uid)

    @pytest.mark.asyncio
    async def test_no_previous_session_no_observation(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=True)
        uid = "user-7"

        # First-ever message — no prev_count key
        with patch(
            "companion_bot_core.orchestrator.orchestrator.SESSION_MESSAGES"
        ) as mock_hist:
            await _track_session_message(redis, uid)
            mock_hist.observe.assert_not_called()


# ---------------------------------------------------------------------------
# Quality metrics in process_message
# ---------------------------------------------------------------------------


class TestQualityMetricsInPipeline:
    @pytest.mark.asyncio
    async def test_response_length_observed(self) -> None:
        """RESPONSE_LENGTH_SENTENCES is observed with the reply's sentence count."""
        user_id = uuid4()
        redis = fakeredis.FakeRedis(decode_responses=True)
        session = _make_session()
        store = InMemorySnapshotStore()
        # Reply with 3 sentences
        reply_text = "First sentence. Second sentence. Third sentence."
        client = _make_mock_client(_make_inference_reply(reply_text))

        with (
            patch(
                "companion_bot_core.orchestrator.orchestrator.classify",
                return_value=_make_detection(action="pass_through"),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.generate_reply",
                return_value=_make_inference_reply(reply_text),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.RESPONSE_LENGTH_SENTENCES"
            ) as mock_hist,
        ):
            await process_message(
                user_id=user_id,
                message_text="Tell me something.",
                session=session,
                snapshot_store=store,
                redis=redis,
                chat_client=client,
            )

        mock_hist.observe.assert_called_once_with(3)

    @pytest.mark.asyncio
    async def test_farewell_counter_incremented(self) -> None:
        """FAREWELL_DETECTED is incremented when emotion mode is farewell."""
        user_id = uuid4()
        redis = fakeredis.FakeRedis(decode_responses=True)
        session = _make_session()
        store = InMemorySnapshotStore()
        client = _make_mock_client(_make_inference_reply("Bye!"))

        with (
            patch(
                "companion_bot_core.orchestrator.orchestrator.classify",
                return_value=_make_detection(action="pass_through"),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.generate_reply",
                return_value=_make_inference_reply("Bye!"),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.detect_emotion",
                return_value=MagicMock(mode="farewell", confidence=0.7),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.FAREWELL_DETECTED"
            ) as mock_ctr,
        ):
            await process_message(
                user_id=user_id,
                message_text="пока!",
                session=session,
                snapshot_store=store,
                redis=redis,
                chat_client=client,
            )

        mock_ctr.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_farewell_not_incremented_on_neutral(self) -> None:
        """FAREWELL_DETECTED is NOT incremented for non-farewell messages."""
        user_id = uuid4()
        redis = fakeredis.FakeRedis(decode_responses=True)
        session = _make_session()
        store = InMemorySnapshotStore()
        client = _make_mock_client(_make_inference_reply("Hi!"))

        with (
            patch(
                "companion_bot_core.orchestrator.orchestrator.classify",
                return_value=_make_detection(action="pass_through"),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.generate_reply",
                return_value=_make_inference_reply("Hi!"),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.detect_emotion",
                return_value=MagicMock(mode="neutral", confidence=0.0),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.FAREWELL_DETECTED"
            ) as mock_ctr,
        ):
            await process_message(
                user_id=user_id,
                message_text="hello",
                session=session,
                snapshot_store=store,
                redis=redis,
                chat_client=client,
            )

        mock_ctr.inc.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_tracking_called_on_normal_message(self) -> None:
        """_track_session_message is called for normal chat."""
        user_id = uuid4()
        redis = fakeredis.FakeRedis(decode_responses=True)
        session = _make_session()
        store = InMemorySnapshotStore()
        client = _make_mock_client(_make_inference_reply("Reply."))

        with (
            patch(
                "companion_bot_core.orchestrator.orchestrator.classify",
                return_value=_make_detection(action="pass_through"),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.generate_reply",
                return_value=_make_inference_reply("Reply."),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator._track_session_message",
                new_callable=AsyncMock,
            ) as mock_track,
        ):
            await process_message(
                user_id=user_id,
                message_text="hi",
                session=session,
                snapshot_store=store,
                redis=redis,
                chat_client=client,
            )

        mock_track.assert_called_once()
        call_kwargs = mock_track.call_args
        assert call_kwargs[1]["is_farewell"] is False

    @pytest.mark.asyncio
    async def test_session_tracking_farewell_flag_on_farewell(self) -> None:
        """_track_session_message is called with is_farewell=True on farewell."""
        user_id = uuid4()
        redis = fakeredis.FakeRedis(decode_responses=True)
        session = _make_session()
        store = InMemorySnapshotStore()
        client = _make_mock_client(_make_inference_reply("Bye!"))

        with (
            patch(
                "companion_bot_core.orchestrator.orchestrator.classify",
                return_value=_make_detection(action="pass_through"),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.generate_reply",
                return_value=_make_inference_reply("Bye!"),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator.detect_emotion",
                return_value=MagicMock(mode="farewell", confidence=0.7),
            ),
            patch(
                "companion_bot_core.orchestrator.orchestrator._track_session_message",
                new_callable=AsyncMock,
            ) as mock_track,
        ):
            await process_message(
                user_id=user_id,
                message_text="пока!",
                session=session,
                snapshot_store=store,
                redis=redis,
                chat_client=client,
            )

        mock_track.assert_called_once()
        call_kwargs = mock_track.call_args
        assert call_kwargs[1]["is_farewell"] is True
