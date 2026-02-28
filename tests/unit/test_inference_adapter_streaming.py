"""Unit tests for generate_reply_stream (inference adapter streaming path)."""

from __future__ import annotations

from unittest.mock import AsyncMock

from companion_bot_core.inference.adapter import generate_reply_stream
from companion_bot_core.inference.client import ChatAPIClient
from companion_bot_core.inference.schemas import (
    ChatMessage,
    InferenceReply,
    OpenAIResponse,
    UserContext,
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


def _make_openai_response(
    content: str = "Streamed reply",
    finish_reason: str = "stop",
    refusal: str | None = None,
    prompt_tokens: int = 20,
    completion_tokens: int = 5,
) -> OpenAIResponse:
    raw = {
        "id": "chatcmpl-stream",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": refusal,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return OpenAIResponse.model_validate(raw)


def _make_mock_client(response: OpenAIResponse) -> ChatAPIClient:
    mock = AsyncMock(spec=ChatAPIClient)
    mock.chat_completion_stream = AsyncMock(return_value=response)
    return mock


# ---------------------------------------------------------------------------
# generate_reply_stream — basic behaviour
# ---------------------------------------------------------------------------


async def test_generate_reply_stream_returns_inference_reply() -> None:
    client = _make_mock_client(_make_openai_response())
    chunks: list[str] = []

    async def on_delta(chunk: str) -> None:
        chunks.append(chunk)

    result = await generate_reply_stream(client, _BASE_CONTEXT, "What is 2+2?", on_delta)
    assert isinstance(result, InferenceReply)
    assert result.reply == "Streamed reply"


async def test_generate_reply_stream_passes_on_delta_to_client() -> None:
    client = _make_mock_client(_make_openai_response())
    marker: list[str] = []

    async def on_delta(chunk: str) -> None:
        marker.append(chunk)

    await generate_reply_stream(client, _BASE_CONTEXT, "hello", on_delta)

    client.chat_completion_stream.assert_called_once()  # type: ignore[attr-defined]
    call_kwargs = client.chat_completion_stream.call_args[1]  # type: ignore[attr-defined]
    assert call_kwargs["on_delta"] is on_delta


async def test_generate_reply_stream_assembles_messages_correctly() -> None:
    client = _make_mock_client(_make_openai_response())

    async def noop(chunk: str) -> None:
        pass

    await generate_reply_stream(client, _BASE_CONTEXT, "New question", noop)

    call_args = client.chat_completion_stream.call_args  # type: ignore[attr-defined]
    messages: list[ChatMessage] = call_args[0][0]

    assert messages[0].role == "system"
    assert messages[0].content == "You are a helpful companion."
    assert messages[-1].role == "user"
    assert messages[-1].content == "New question"


async def test_generate_reply_stream_passes_max_tokens() -> None:
    ctx = UserContext(user_id="u1", system_prompt="s", max_tokens=512)
    client = _make_mock_client(_make_openai_response())

    async def noop(chunk: str) -> None:
        pass

    await generate_reply_stream(client, ctx, "hi", noop)

    call_kwargs = client.chat_completion_stream.call_args[1]  # type: ignore[attr-defined]
    assert call_kwargs["max_tokens"] == 512


# ---------------------------------------------------------------------------
# Safety flags
# ---------------------------------------------------------------------------


async def test_generate_reply_stream_safety_flags_content_filter() -> None:
    client = _make_mock_client(_make_openai_response(content="", finish_reason="content_filter"))

    async def noop(chunk: str) -> None:
        pass

    result = await generate_reply_stream(client, _BASE_CONTEXT, "bad", noop)
    assert result.safety_flags.content_filtered is True
    assert result.safety_flags.finish_reason == "content_filter"


async def test_generate_reply_stream_safety_flags_refusal() -> None:
    client = _make_mock_client(
        _make_openai_response(content=None, refusal="I can't help.")  # type: ignore[arg-type]
    )

    async def noop(chunk: str) -> None:
        pass

    result = await generate_reply_stream(client, _BASE_CONTEXT, "problem", noop)
    assert result.safety_flags.refusal is True


async def test_generate_reply_stream_token_usage() -> None:
    client = _make_mock_client(
        _make_openai_response(prompt_tokens=30, completion_tokens=10)
    )

    async def noop(chunk: str) -> None:
        pass

    result = await generate_reply_stream(client, _BASE_CONTEXT, "tokens", noop)
    assert result.usage.prompt_tokens == 30
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 40
