"""Unit tests for companion_bot_core.inference.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from companion_bot_core.inference.schemas import (
    ChatMessage,
    InferenceReply,
    OpenAIResponse,
    SafetyFlags,
    TokenUsage,
    UserContext,
)

# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


def test_chat_message_valid_roles() -> None:
    for role in ("system", "user", "assistant"):
        msg = ChatMessage(role=role, content="hello")
        assert msg.role == role
        assert msg.content == "hello"


def test_chat_message_invalid_role() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="human", content="hello")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# UserContext
# ---------------------------------------------------------------------------


def test_user_context_defaults() -> None:
    ctx = UserContext(user_id="u1", system_prompt="You are helpful.")
    assert ctx.conversation_history == []
    assert ctx.max_tokens == 2048


def test_user_context_with_history() -> None:
    history = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
    ]
    ctx = UserContext(
        user_id="u1",
        system_prompt="You are helpful.",
        conversation_history=history,
    )
    assert len(ctx.conversation_history) == 2


def test_user_context_max_tokens_bounds() -> None:
    with pytest.raises(ValidationError):
        UserContext(user_id="u1", system_prompt="s", max_tokens=0)
    with pytest.raises(ValidationError):
        UserContext(user_id="u1", system_prompt="s", max_tokens=20000)


# ---------------------------------------------------------------------------
# OpenAIResponse (internal schema) — validates raw provider JSON
# ---------------------------------------------------------------------------

_VALID_RAW = {
    "id": "chatcmpl-abc123",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!", "refusal": None},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
}


def test_openai_response_valid() -> None:
    resp = OpenAIResponse.model_validate(_VALID_RAW)
    assert resp.id == "chatcmpl-abc123"
    assert len(resp.choices) == 1
    assert resp.choices[0].message.content == "Hello!"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.total_tokens == 15


def test_openai_response_requires_at_least_one_choice() -> None:
    raw = {**_VALID_RAW, "choices": []}
    with pytest.raises(ValidationError):
        OpenAIResponse.model_validate(raw)


def test_openai_response_null_content_allowed() -> None:
    raw = {
        **_VALID_RAW,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": "I cannot help with that.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    resp = OpenAIResponse.model_validate(raw)
    assert resp.choices[0].message.content is None
    assert resp.choices[0].message.refusal == "I cannot help with that."


def test_openai_response_content_filter_finish_reason() -> None:
    raw = {
        **_VALID_RAW,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "content_filter",
            }
        ],
    }
    resp = OpenAIResponse.model_validate(raw)
    assert resp.choices[0].finish_reason == "content_filter"


def test_openai_response_missing_refusal_field() -> None:
    """refusal is optional — omitting it should parse fine."""
    raw = {
        **_VALID_RAW,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }
        ],
    }
    resp = OpenAIResponse.model_validate(raw)
    assert resp.choices[0].message.refusal is None


# ---------------------------------------------------------------------------
# InferenceReply
# ---------------------------------------------------------------------------


def test_inference_reply_construction() -> None:
    reply = InferenceReply(
        reply="Hi there!",
        usage=TokenUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
        safety_flags=SafetyFlags(
            content_filtered=False, refusal=False, finish_reason="stop"
        ),
    )
    assert reply.reply == "Hi there!"
    assert reply.usage.total_tokens == 11
    assert reply.safety_flags.content_filtered is False
