"""Unit tests for the SSE streaming pipeline (fake adapter → adapter → orchestrator)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from companion_bot_core.dev.fake_client import FakeChatAPIClient
from companion_bot_core.inference.adapter import generate_reply_stream
from companion_bot_core.inference.schemas import (
    ChatMessage,
    InferenceReply,
    UserContext,
    _StreamEnd,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONTEXT = UserContext(
    user_id="user-42",
    system_prompt="You are a helpful companion.",
    conversation_history=[
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
    ],
)


# ---------------------------------------------------------------------------
# FakeChatAPIClient.chat_completion_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_stream_yields_chunks_then_sentinel() -> None:
    """The fake streaming method yields str chunks followed by _StreamEnd."""
    client = FakeChatAPIClient()
    messages = [
        ChatMessage(role="system", content="You are a helper."),
        ChatMessage(role="user", content="Say hi"),
    ]

    chunks: list[str] = []
    sentinel: _StreamEnd | None = None
    async for item in client.chat_completion_stream(messages):
        if isinstance(item, _StreamEnd):
            sentinel = item
        else:
            chunks.append(item)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "[Dev mode]" in full_text
    assert "Say hi" in full_text
    assert sentinel is not None
    assert sentinel.finish_reason == "stop"
    assert sentinel.refusal is False
    assert sentinel.total_tokens == 20


@pytest.mark.asyncio
async def test_fake_stream_no_user_message_falls_back() -> None:
    """When no user turn is present, falls back to 'Hello!'."""
    client = FakeChatAPIClient()
    messages = [ChatMessage(role="system", content="System only.")]

    chunks: list[str] = []
    async for item in client.chat_completion_stream(messages):
        if isinstance(item, str):
            chunks.append(item)

    assert "Hello!" in "".join(chunks)


# ---------------------------------------------------------------------------
# generate_reply_stream (adapter layer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_reply_stream_returns_inference_reply() -> None:
    """generate_reply_stream returns an InferenceReply with the full text."""
    client = FakeChatAPIClient()
    on_chunk = AsyncMock()

    result = await generate_reply_stream(
        client, _BASE_CONTEXT, "What is 2+2?", on_chunk=on_chunk,
    )

    assert isinstance(result, InferenceReply)
    assert "[Dev mode]" in result.reply
    assert "What is 2+2?" in result.reply


@pytest.mark.asyncio
async def test_generate_reply_stream_calls_on_chunk() -> None:
    """on_chunk is called for each content delta."""
    client = FakeChatAPIClient()
    received: list[str] = []

    async def capture(chunk: str) -> None:
        received.append(chunk)

    result = await generate_reply_stream(
        client, _BASE_CONTEXT, "Tell me a joke", on_chunk=capture,
    )

    assert len(received) > 0
    assert "".join(received) == result.reply


@pytest.mark.asyncio
async def test_generate_reply_stream_safety_flags() -> None:
    """Safety flags are populated from the _StreamEnd sentinel."""
    client = FakeChatAPIClient()
    result = await generate_reply_stream(
        client, _BASE_CONTEXT, "hi", on_chunk=AsyncMock(),
    )

    assert result.safety_flags.finish_reason == "stop"
    assert result.safety_flags.refusal is False
    assert result.safety_flags.content_filtered is False


@pytest.mark.asyncio
async def test_generate_reply_stream_token_usage() -> None:
    """Token usage is populated from the _StreamEnd sentinel."""
    client = FakeChatAPIClient()
    result = await generate_reply_stream(
        client, _BASE_CONTEXT, "hi", on_chunk=AsyncMock(),
    )

    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 20


@pytest.mark.asyncio
async def test_generate_reply_stream_assembles_messages_correctly() -> None:
    """Verify the message list passed to the client is system + history + user."""
    mock_client = AsyncMock(spec=FakeChatAPIClient)

    # Create a real fake client to get genuine stream output
    real_client = FakeChatAPIClient()

    async def fake_stream(
        messages: list[ChatMessage], max_tokens: int = 1024, temperature: float = 0.7,
    ):  # type: ignore[override]
        # Record the messages for assertions
        mock_client._recorded_messages = messages
        async for item in real_client.chat_completion_stream(messages, max_tokens, temperature):
            yield item

    mock_client.chat_completion_stream = fake_stream

    await generate_reply_stream(
        mock_client, _BASE_CONTEXT, "New question", on_chunk=AsyncMock(),
    )

    messages = mock_client._recorded_messages
    assert messages[0].role == "system"
    assert messages[0].content == "You are a helpful companion."
    assert messages[1].role == "user"
    assert messages[1].content == "Hello"
    assert messages[2].role == "assistant"
    assert messages[2].content == "Hi there!"
    assert messages[3].role == "user"
    assert messages[3].content == "New question"
