"""Unit tests for companion_bot_core.inference.adapter (generate_reply)."""

from __future__ import annotations

from unittest.mock import AsyncMock

from companion_bot_core.inference.adapter import generate_reply
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
