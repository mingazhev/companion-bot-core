"""Unit tests for companion_bot_core.inference.adapter (generate_reply)."""

from __future__ import annotations

from unittest.mock import AsyncMock

from companion_bot_core.inference.adapter import generate_reply, generate_reply_stream
from companion_bot_core.inference.client import ChatAPIClient
from companion_bot_core.inference.schemas import (
    ChatMessage,
    InferenceReply,
    OpenAIResponse,
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


def _make_openai_response(
    content: str = "Sure, I can help!",
    finish_reason: str = "stop",
    refusal: str | None = None,
    prompt_tokens: int = 20,
    completion_tokens: int = 8,
) -> OpenAIResponse:
    raw = {
        "id": "chatcmpl-test",
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
    mock.chat_completion = AsyncMock(return_value=response)
    return mock


async def _async_iter(items: list[str | _StreamEnd]):  # type: ignore[return]
    """Yield items from a list as an async iterator."""
    for item in items:
        yield item


def _make_stream_client(
    chunks: list[str],
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> ChatAPIClient:
    """Return a mock client whose chat_completion_stream yields the given chunks."""
    items: list[str | _StreamEnd] = list(chunks) + [
        _StreamEnd(
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    ]
    mock = AsyncMock(spec=ChatAPIClient)
    mock.chat_completion_stream.return_value = _async_iter(items)
    return mock


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


async def test_generate_reply_returns_inference_reply() -> None:
    client = _make_mock_client(_make_openai_response())
    result = await generate_reply(client, _BASE_CONTEXT, "What is 2+2?")
    assert isinstance(result, InferenceReply)
    assert result.reply == "Sure, I can help!"


async def test_generate_reply_token_usage_populated() -> None:
    client = _make_mock_client(
        _make_openai_response(prompt_tokens=15, completion_tokens=6)
    )
    result = await generate_reply(client, _BASE_CONTEXT, "Tell me a joke.")
    assert result.usage.prompt_tokens == 15
    assert result.usage.completion_tokens == 6
    assert result.usage.total_tokens == 21


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------


async def test_generate_reply_assembles_messages_correctly() -> None:
    """Verify the system + history + new user turn ordering."""
    client = _make_mock_client(_make_openai_response())
    await generate_reply(client, _BASE_CONTEXT, "New question")

    client.chat_completion.assert_called_once()  # type: ignore[attr-defined]
    call_args = client.chat_completion.call_args  # type: ignore[attr-defined]
    messages: list[ChatMessage] = call_args[0][0]

    assert messages[0].role == "system"
    assert messages[0].content == "You are a helpful companion."
    assert messages[1].role == "user"
    assert messages[1].content == "Hello"
    assert messages[2].role == "assistant"
    assert messages[2].content == "Hi there!"
    assert messages[3].role == "user"
    assert messages[3].content == "New question"


async def test_generate_reply_no_history() -> None:
    ctx = UserContext(user_id="u1", system_prompt="You are helpful.")
    client = _make_mock_client(_make_openai_response())
    await generate_reply(client, ctx, "Start fresh")

    messages: list[ChatMessage] = client.chat_completion.call_args[0][0]  # type: ignore[attr-defined]
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[1].content == "Start fresh"


async def test_generate_reply_passes_max_tokens() -> None:
    ctx = UserContext(user_id="u1", system_prompt="s", max_tokens=512)
    client = _make_mock_client(_make_openai_response())
    await generate_reply(client, ctx, "hi")

    call_args = client.chat_completion.call_args  # type: ignore[attr-defined]
    assert call_args[1]["max_tokens"] == 512


# ---------------------------------------------------------------------------
# Safety flags
# ---------------------------------------------------------------------------


async def test_safety_flags_normal_response() -> None:
    client = _make_mock_client(_make_openai_response(finish_reason="stop"))
    result = await generate_reply(client, _BASE_CONTEXT, "hey")
    assert result.safety_flags.content_filtered is False
    assert result.safety_flags.refusal is False
    assert result.safety_flags.finish_reason == "stop"


async def test_safety_flags_content_filter() -> None:
    client = _make_mock_client(
        _make_openai_response(content="", finish_reason="content_filter")
    )
    result = await generate_reply(client, _BASE_CONTEXT, "bad request")
    assert result.safety_flags.content_filtered is True
    assert result.safety_flags.finish_reason == "content_filter"


async def test_safety_flags_refusal() -> None:
    client = _make_mock_client(
        _make_openai_response(
            content=None,  # type: ignore[arg-type]
            finish_reason="stop",
            refusal="I'm sorry, I can't help with that.",
        )
    )
    result = await generate_reply(client, _BASE_CONTEXT, "problematic request")
    assert result.safety_flags.refusal is True


async def test_safety_flags_null_finish_reason_defaults_to_stop() -> None:
    """finish_reason=None from provider should default to 'stop'."""
    raw = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": None,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    client = _make_mock_client(OpenAIResponse.model_validate(raw))
    result = await generate_reply(client, _BASE_CONTEXT, "hey")
    assert result.safety_flags.finish_reason == "stop"


# ---------------------------------------------------------------------------
# Empty reply
# ---------------------------------------------------------------------------


async def test_generate_reply_empty_content_returns_empty_string() -> None:
    """Null content from the provider should produce an empty reply string."""
    raw = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "content_filter",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }
    client = _make_mock_client(OpenAIResponse.model_validate(raw))
    result = await generate_reply(client, _BASE_CONTEXT, "bad")
    assert result.reply == ""


# ---------------------------------------------------------------------------
# generate_reply_stream — basic behaviour
# ---------------------------------------------------------------------------


async def test_generate_reply_stream_returns_inference_reply() -> None:
    client = _make_stream_client(["Hello", " there", "!"])
    chunks_received: list[str] = []

    async def on_chunk(chunk: str) -> None:
        chunks_received.append(chunk)

    result = await generate_reply_stream(client, _BASE_CONTEXT, "hi", on_chunk=on_chunk)

    assert isinstance(result, InferenceReply)
    assert result.reply == "Hello there!"


async def test_generate_reply_stream_callback_receives_all_chunks() -> None:
    expected = ["Hello", " world", "!"]
    client = _make_stream_client(expected)
    received: list[str] = []

    async def on_chunk(chunk: str) -> None:
        received.append(chunk)

    await generate_reply_stream(client, _BASE_CONTEXT, "test", on_chunk=on_chunk)
    assert received == expected


async def _noop_chunk(chunk: str) -> None:  # noqa: ARG001
    """Async no-op callback for use in tests that don't need to inspect chunks."""


async def test_generate_reply_stream_usage_from_sentinel() -> None:
    client = _make_stream_client(
        ["hi"], prompt_tokens=15, completion_tokens=3
    )
    received: list[str] = []

    async def collect(chunk: str) -> None:
        received.append(chunk)

    result = await generate_reply_stream(
        client, _BASE_CONTEXT, "test", on_chunk=collect
    )
    assert result.usage.prompt_tokens == 15
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 18


async def test_generate_reply_stream_content_filter_finish_reason() -> None:
    client = _make_stream_client(["..."], finish_reason="content_filter")

    result = await generate_reply_stream(
        client, _BASE_CONTEXT, "bad", on_chunk=_noop_chunk
    )
    assert result.safety_flags.content_filtered is True
    assert result.safety_flags.finish_reason == "content_filter"


async def test_generate_reply_stream_no_sentinel_zeros_usage() -> None:
    """When no _StreamEnd sentinel arrives, token counts fall back to zero."""

    async def _stream_no_sentinel():  # type: ignore[return]
        yield "hello"

    mock = AsyncMock(spec=ChatAPIClient)
    mock.chat_completion_stream.return_value = _stream_no_sentinel()

    result = await generate_reply_stream(
        mock, _BASE_CONTEXT, "hi", on_chunk=_noop_chunk
    )
    assert result.reply == "hello"
    assert result.usage.total_tokens == 0
    assert result.safety_flags.finish_reason == "stop"


async def test_generate_reply_stream_refusal_flag_is_false() -> None:
    """Streaming path always sets refusal=False (no refusal field in deltas)."""
    client = _make_stream_client(["ok"])
    result = await generate_reply_stream(
        client, _BASE_CONTEXT, "hey", on_chunk=_noop_chunk
    )
    assert result.safety_flags.refusal is False


async def test_generate_reply_stream_passes_max_tokens() -> None:
    ctx = UserContext(user_id="u1", system_prompt="s", max_tokens=256)
    client = _make_stream_client(["hi"])

    await generate_reply_stream(
        client, ctx, "hello", on_chunk=_noop_chunk
    )
    client.chat_completion_stream.assert_called_once()  # type: ignore[attr-defined]
    call_args = client.chat_completion_stream.call_args  # type: ignore[attr-defined]
    assert call_args[1]["max_tokens"] == 256

