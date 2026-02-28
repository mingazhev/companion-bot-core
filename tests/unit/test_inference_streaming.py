"""Unit tests for streaming inference code paths.

Covers:
- ChatAPIClient._raw_completion_stream / chat_completion_stream
- generate_reply_stream
- FakeChatAPIClient.chat_completion_stream
- process_message with on_partial_reply
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from companion_bot_core.dev.fake_client import FakeChatAPIClient
from companion_bot_core.inference.adapter import generate_reply_stream
from companion_bot_core.inference.client import ChatAPIClient
from companion_bot_core.inference.schemas import (
    ChatMessage,
    InferenceReply,
    OpenAIResponse,
    SafetyFlags,
    TokenUsage,
    UserContext,
)
from companion_bot_core.orchestrator.orchestrator import process_message
from companion_bot_core.prompt.snapshot_store import InMemorySnapshotStore

import fakeredis.aioredis as fakeredis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system(content: str) -> ChatMessage:
    return ChatMessage(role="system", content=content)


def _user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _make_inference_reply(text: str = "Hello!", total_tokens: int = 15) -> InferenceReply:
    return InferenceReply(
        reply=text,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=total_tokens),
        safety_flags=SafetyFlags(content_filtered=False, refusal=False, finish_reason="stop"),
    )


def _make_openai_response(content: str = "Hello!") -> OpenAIResponse:
    raw = {
        "id": "chatcmpl-stream-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return OpenAIResponse.model_validate(raw)


def _make_session() -> AsyncMock:
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=ctx)
    session.info = {}
    return session


_BASE_CONTEXT = UserContext(
    user_id="user-stream-test",
    system_prompt="You are a helpful companion.",
    conversation_history=[],
)


# ---------------------------------------------------------------------------
# FakeChatAPIClient.chat_completion_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_client_stream_collects_deltas() -> None:
    """All word chunks emitted by the fake client must reconstruct the reply."""
    client = FakeChatAPIClient()
    deltas: list[str] = []

    async def capture(delta: str) -> None:
        deltas.append(delta)

    response = await client.chat_completion_stream([_user("Hello")], on_delta=capture)

    assert isinstance(response, OpenAIResponse)
    assert deltas, "on_delta should have been called at least once"
    assert "".join(deltas) == response.choices[0].message.content


@pytest.mark.asyncio
async def test_fake_client_stream_echo_reply() -> None:
    """Streaming response should echo the last user message like chat_completion."""
    client = FakeChatAPIClient()
    received: list[str] = []

    async def capture(delta: str) -> None:
        received.append(delta)

    response = await client.chat_completion_stream(
        [_user("What is 2+2?")],
        on_delta=capture,
    )

    full = response.choices[0].message.content or ""
    assert "What is 2+2?" in full
    assert full.startswith("[Dev mode]")


@pytest.mark.asyncio
async def test_fake_client_stream_refinement_path() -> None:
    """Refinement marker should trigger JSON response via streaming path too."""
    client = FakeChatAPIClient()
    deltas: list[str] = []

    async def capture(delta: str) -> None:
        deltas.append(delta)

    response = await client.chat_completion_stream(
        [
            _system("You are a prompt-refinement assistant."),
            _user("Refine this."),
        ],
        on_delta=capture,
    )

    content = response.choices[0].message.content or ""
    data = json.loads(content)
    assert "proposed_delta" in data
    assert deltas  # on_delta was called


@pytest.mark.asyncio
async def test_fake_client_stream_is_valid_openai_response() -> None:
    """Streaming response must validate as OpenAIResponse."""
    client = FakeChatAPIClient()

    async def noop(delta: str) -> None:
        pass

    response = await client.chat_completion_stream(
        [_user("Test")],
        on_delta=noop,
    )
    assert isinstance(response, OpenAIResponse)
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_fake_client_stream_accepts_max_tokens_and_temperature() -> None:
    client = FakeChatAPIClient()

    async def noop(delta: str) -> None:
        pass

    response = await client.chat_completion_stream(
        [_user("hi")],
        on_delta=noop,
        max_tokens=256,
        temperature=0.0,
    )
    assert isinstance(response, OpenAIResponse)


# ---------------------------------------------------------------------------
# generate_reply_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_reply_stream_returns_inference_reply() -> None:
    """generate_reply_stream should return InferenceReply."""
    mock_client = AsyncMock(spec=ChatAPIClient)
    mock_client.chat_completion_stream = AsyncMock(
        return_value=_make_openai_response("Streaming reply!")
    )

    deltas: list[str] = []

    async def capture(delta: str) -> None:
        deltas.append(delta)

    result = await generate_reply_stream(mock_client, _BASE_CONTEXT, "hi", capture)

    assert isinstance(result, InferenceReply)
    assert result.reply == "Streaming reply!"


@pytest.mark.asyncio
async def test_generate_reply_stream_passes_on_partial_reply_as_on_delta() -> None:
    """generate_reply_stream must forward on_partial_reply as the on_delta kwarg."""
    mock_client = AsyncMock(spec=ChatAPIClient)
    mock_client.chat_completion_stream = AsyncMock(
        return_value=_make_openai_response("Hello!")
    )

    sentinel: list[bool] = []

    async def my_callback(delta: str) -> None:
        sentinel.append(True)

    await generate_reply_stream(mock_client, _BASE_CONTEXT, "test", my_callback)

    mock_client.chat_completion_stream.assert_called_once()
    call_kwargs = mock_client.chat_completion_stream.call_args[1]
    assert call_kwargs["on_delta"] is my_callback


@pytest.mark.asyncio
async def test_generate_reply_stream_assembles_messages_correctly() -> None:
    """Message list must follow system + history + new user turn order."""
    mock_client = AsyncMock(spec=ChatAPIClient)
    mock_client.chat_completion_stream = AsyncMock(
        return_value=_make_openai_response("ok")
    )

    ctx = UserContext(
        user_id="u1",
        system_prompt="System prompt.",
        conversation_history=[
            ChatMessage(role="user", content="Prior question"),
            ChatMessage(role="assistant", content="Prior answer"),
        ],
    )

    async def noop(delta: str) -> None:
        pass

    await generate_reply_stream(mock_client, ctx, "New question", noop)

    messages: list[ChatMessage] = mock_client.chat_completion_stream.call_args[0][0]
    assert messages[0].role == "system"
    assert messages[0].content == "System prompt."
    assert messages[1].content == "Prior question"
    assert messages[2].content == "Prior answer"
    assert messages[3].role == "user"
    assert messages[3].content == "New question"


@pytest.mark.asyncio
async def test_generate_reply_stream_passes_max_tokens() -> None:
    mock_client = AsyncMock(spec=ChatAPIClient)
    mock_client.chat_completion_stream = AsyncMock(
        return_value=_make_openai_response("ok")
    )
    ctx = UserContext(user_id="u1", system_prompt="s", max_tokens=512)

    async def noop(delta: str) -> None:
        pass

    await generate_reply_stream(mock_client, ctx, "hi", noop)

    kwargs = mock_client.chat_completion_stream.call_args[1]
    assert kwargs["max_tokens"] == 512


@pytest.mark.asyncio
async def test_generate_reply_stream_safety_flags_content_filter() -> None:
    raw = {
        "id": "x",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "", "refusal": None},
                "finish_reason": "content_filter",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }
    mock_client = AsyncMock(spec=ChatAPIClient)
    mock_client.chat_completion_stream = AsyncMock(
        return_value=OpenAIResponse.model_validate(raw)
    )

    async def noop(delta: str) -> None:
        pass

    result = await generate_reply_stream(mock_client, _BASE_CONTEXT, "bad", noop)
    assert result.safety_flags.content_filtered is True


# ---------------------------------------------------------------------------
# process_message with on_partial_reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_message_streaming_calls_on_partial_reply() -> None:
    """When on_partial_reply is provided, generate_reply_stream is used."""
    deltas: list[str] = []

    async def capture(delta: str) -> None:
        deltas.append(delta)

    redis = fakeredis.FakeRedis()
    snapshot_store = InMemorySnapshotStore()
    session = _make_session()
    user_id = uuid4()

    with (
        patch(
            "companion_bot_core.orchestrator.orchestrator.generate_reply_stream",
            new_callable=AsyncMock,
        ) as mock_stream,
        patch(
            "companion_bot_core.orchestrator.orchestrator.classify",
        ) as mock_classify,
        patch(
            "companion_bot_core.orchestrator.orchestrator.load_user_context",
            new_callable=AsyncMock,
        ) as mock_load,
        patch(
            "companion_bot_core.orchestrator.orchestrator.is_user_abuse_blocked",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator.check_prompt_injection",
            return_value=MagicMock(allowed=True),
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator.check_unsafe_role_change",
            return_value=MagicMock(allowed=True),
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator.check_risky_capability",
            return_value=MagicMock(allowed=True),
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator._persist_messages",
            new_callable=AsyncMock,
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator._maybe_enqueue_refinement",
            new_callable=AsyncMock,
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator.enqueue_if_cadence_due",
            new_callable=AsyncMock,
        ),
    ):
        mock_classify.return_value = MagicMock(
            intent="normal_chat",
            action="pass_through",
            risk_level="low",
            confidence=0.9,
            clarification_question=None,
        )
        mock_load.return_value = _BASE_CONTEXT
        mock_stream.return_value = _make_inference_reply("Streamed reply")

        result = await process_message(
            user_id=user_id,
            message_text="Hello",
            session=session,
            snapshot_store=snapshot_store,
            redis=redis,
            chat_client=AsyncMock(),
            on_partial_reply=capture,
        )

    assert result == "Streamed reply"
    mock_stream.assert_called_once()


@pytest.mark.asyncio
async def test_process_message_without_on_partial_reply_uses_generate_reply() -> None:
    """Without on_partial_reply, the non-streaming generate_reply is used."""
    redis = fakeredis.FakeRedis()
    snapshot_store = InMemorySnapshotStore()
    session = _make_session()
    user_id = uuid4()

    with (
        patch(
            "companion_bot_core.orchestrator.orchestrator.generate_reply",
            new_callable=AsyncMock,
        ) as mock_reply,
        patch(
            "companion_bot_core.orchestrator.orchestrator.generate_reply_stream",
            new_callable=AsyncMock,
        ) as mock_stream,
        patch(
            "companion_bot_core.orchestrator.orchestrator.classify",
        ) as mock_classify,
        patch(
            "companion_bot_core.orchestrator.orchestrator.load_user_context",
            new_callable=AsyncMock,
        ) as mock_load,
        patch(
            "companion_bot_core.orchestrator.orchestrator.is_user_abuse_blocked",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator.check_prompt_injection",
            return_value=MagicMock(allowed=True),
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator.check_unsafe_role_change",
            return_value=MagicMock(allowed=True),
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator.check_risky_capability",
            return_value=MagicMock(allowed=True),
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator._persist_messages",
            new_callable=AsyncMock,
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator._maybe_enqueue_refinement",
            new_callable=AsyncMock,
        ),
        patch(
            "companion_bot_core.orchestrator.orchestrator.enqueue_if_cadence_due",
            new_callable=AsyncMock,
        ),
    ):
        mock_classify.return_value = MagicMock(
            intent="normal_chat",
            action="pass_through",
            risk_level="low",
            confidence=0.9,
            clarification_question=None,
        )
        mock_load.return_value = _BASE_CONTEXT
        mock_reply.return_value = _make_inference_reply("Non-streaming reply")

        result = await process_message(
            user_id=user_id,
            message_text="Hello",
            session=session,
            snapshot_store=snapshot_store,
            redis=redis,
            chat_client=AsyncMock(),
        )

    assert result == "Non-streaming reply"
    mock_reply.assert_called_once()
    mock_stream.assert_not_called()
