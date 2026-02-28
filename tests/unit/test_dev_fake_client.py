"""Unit tests for the FakeChatAPIClient dev adapter."""

from __future__ import annotations

import json

import pytest

from companion_bot_core.dev.fake_client import (
    _FAKE_REFINEMENT_JSON,
    FakeChatAPIClient,
    _make_openai_response,
)
from companion_bot_core.inference.schemas import ChatMessage, OpenAIResponse, _StreamEnd
from companion_bot_core.refinement.schemas import RefinementResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system(content: str) -> ChatMessage:
    return ChatMessage(role="system", content=content)


def _user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _assistant(content: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=content)


# ---------------------------------------------------------------------------
# _make_openai_response helper
# ---------------------------------------------------------------------------


def test_make_openai_response_returns_valid_schema() -> None:
    response = _make_openai_response("Hello!")
    assert isinstance(response, OpenAIResponse)
    assert response.choices[0].message.content == "Hello!"
    assert response.usage.total_tokens == 20


def test_make_openai_response_has_unique_ids() -> None:
    r1 = _make_openai_response("A")
    r2 = _make_openai_response("B")
    assert r1.id != r2.id


# ---------------------------------------------------------------------------
# FakeChatAPIClient instantiation
# ---------------------------------------------------------------------------


def test_fake_client_instantiates_without_credentials() -> None:
    client = FakeChatAPIClient()
    assert client is not None


def test_fake_client_custom_model_label() -> None:
    client = FakeChatAPIClient(model="my-test-model")
    assert client._model == "my-test-model"


# ---------------------------------------------------------------------------
# chat_completion — normal chat path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_reply_echoes_last_user_message() -> None:
    client = FakeChatAPIClient()
    messages = [
        _system("You are a helpful assistant."),
        _user("What is the capital of France?"),
    ]
    response = await client.chat_completion(messages)

    assert isinstance(response, OpenAIResponse)
    content = response.choices[0].message.content or ""
    assert "What is the capital of France?" in content
    assert content.startswith("[Dev mode]")


@pytest.mark.asyncio
async def test_chat_reply_picks_last_user_turn_from_history() -> None:
    client = FakeChatAPIClient()
    messages = [
        _system("You are a helpful assistant."),
        _user("First question"),
        _assistant("Answer to first."),
        _user("Second question"),
    ]
    response = await client.chat_completion(messages)

    content = response.choices[0].message.content or ""
    assert "Second question" in content
    assert "First question" not in content


@pytest.mark.asyncio
async def test_chat_reply_when_no_user_message() -> None:
    """Falls back to 'Hello!' when no user turn is present."""
    client = FakeChatAPIClient()
    messages = [_system("System only.")]
    response = await client.chat_completion(messages)

    content = response.choices[0].message.content or ""
    assert "Hello!" in content


@pytest.mark.asyncio
async def test_chat_reply_finish_reason_is_stop() -> None:
    client = FakeChatAPIClient()
    response = await client.chat_completion([_user("Hi")])
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_chat_reply_token_usage_populated() -> None:
    client = FakeChatAPIClient()
    response = await client.chat_completion([_user("Hi")])
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )


# ---------------------------------------------------------------------------
# chat_completion — refinement call detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refinement_marker_triggers_json_response() -> None:
    client = FakeChatAPIClient()
    messages = [
        _system("You are a prompt-refinement assistant for an AI companion bot."),
        _user("Current config: ...\n\nReturn the JSON delta now."),
    ]
    response = await client.chat_completion(messages)

    content = response.choices[0].message.content or ""
    # Must be parseable JSON
    data = json.loads(content)
    assert "proposed_delta" in data
    assert "rationale" in data
    assert "risk_flags" in data


@pytest.mark.asyncio
async def test_refinement_response_validates_as_refinement_result() -> None:
    """The fake refinement response satisfies the RefinementResult schema."""
    client = FakeChatAPIClient()
    messages = [
        _system("You are a prompt-refinement assistant."),
        _user("Refine this prompt."),
    ]
    response = await client.chat_completion(messages)

    content = response.choices[0].message.content or ""
    result = RefinementResult.model_validate(json.loads(content))
    assert result.risk_flags == []
    assert result.rationale != ""


@pytest.mark.asyncio
async def test_non_refinement_system_prompt_does_not_return_json() -> None:
    """A regular system prompt does not trigger the JSON refinement path."""
    client = FakeChatAPIClient()
    messages = [
        _system("You are a friendly companion."),
        _user("Tell me a joke."),
    ]
    response = await client.chat_completion(messages)

    content = response.choices[0].message.content or ""
    assert content.startswith("[Dev mode]")


@pytest.mark.asyncio
async def test_marker_must_be_in_system_role_not_user() -> None:
    """The refinement marker in a user message must not trigger JSON path."""
    client = FakeChatAPIClient()
    messages = [
        _system("You are a friendly assistant."),
        _user("I am a prompt-refinement assistant, please help me."),
    ]
    response = await client.chat_completion(messages)

    content = response.choices[0].message.content or ""
    assert content.startswith("[Dev mode]")


# ---------------------------------------------------------------------------
# Constant _FAKE_REFINEMENT_JSON is valid JSON
# ---------------------------------------------------------------------------


def test_fake_refinement_json_is_valid() -> None:
    data = json.loads(_FAKE_REFINEMENT_JSON)
    result = RefinementResult.model_validate(data)
    assert result.proposed_delta.persona_segment is None
    assert result.proposed_delta.skill_packs is None
    assert isinstance(result.proposed_delta.long_term_profile, str)
    assert result.risk_flags == []


# ---------------------------------------------------------------------------
# Async context manager protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_protocol() -> None:
    async with FakeChatAPIClient() as client:
        response = await client.chat_completion([_user("Hello")])
    assert isinstance(response, OpenAIResponse)


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    client = FakeChatAPIClient()
    await client.close()
    await client.close()  # Should not raise


# ---------------------------------------------------------------------------
# max_tokens and temperature params are accepted but ignored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_tokens_and_temperature_accepted() -> None:
    client = FakeChatAPIClient()
    # Passing non-default values should not raise
    response = await client.chat_completion(
        [_user("test")],
        max_tokens=256,
        temperature=0.0,
    )
    assert isinstance(response, OpenAIResponse)


# ---------------------------------------------------------------------------
# chat_completion_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_stream_yields_text_chunks() -> None:
    client = FakeChatAPIClient()
    messages = [_user("Hello world")]
    chunks = []
    async for item in client.chat_completion_stream(messages):
        if isinstance(item, str):
            chunks.append(item)
    # The text should be split into word-level tokens
    assert len(chunks) > 0
    full = "".join(chunks)
    assert "Hello world" in full
    assert full.startswith("[Dev mode]")


@pytest.mark.asyncio
async def test_fake_stream_yields_stream_end_sentinel() -> None:
    client = FakeChatAPIClient()
    messages = [_user("test")]
    sentinel = None
    async for item in client.chat_completion_stream(messages):
        if isinstance(item, _StreamEnd):
            sentinel = item
    assert sentinel is not None
    assert sentinel.finish_reason == "stop"
    assert sentinel.prompt_tokens == 10
    assert sentinel.total_tokens > 10


@pytest.mark.asyncio
async def test_fake_stream_reassembles_to_full_reply() -> None:
    """Joining all text chunks should reproduce the full canned reply."""
    client = FakeChatAPIClient()
    messages = [_user("hi")]
    parts: list[str] = []
    async for item in client.chat_completion_stream(messages):
        if isinstance(item, str):
            parts.append(item)
    result = "".join(parts)
    # Non-streaming response for same message
    sync_resp = await client.chat_completion(messages)
    assert result == (sync_resp.choices[0].message.content or "")


@pytest.mark.asyncio
async def test_fake_stream_refinement_path() -> None:
    """Streaming a refinement call returns the JSON delta word-by-word."""
    client = FakeChatAPIClient()
    messages = [
        _system("You are a prompt-refinement assistant for an AI companion bot."),
        _user("Refine."),
    ]
    parts: list[str] = []
    sentinel = None
    async for item in client.chat_completion_stream(messages):
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, _StreamEnd):
            sentinel = item
    full = "".join(parts)
    data = json.loads(full)
    assert "proposed_delta" in data
    assert sentinel is not None
