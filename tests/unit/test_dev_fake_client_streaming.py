"""Unit tests for FakeChatAPIClient streaming (chat_completion_stream)."""

from __future__ import annotations

from companion_bot_core.dev.fake_client import FakeChatAPIClient
from companion_bot_core.inference.schemas import ChatMessage, OpenAIResponse


def _user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _system(content: str) -> ChatMessage:
    return ChatMessage(role="system", content=content)


# ---------------------------------------------------------------------------
# chat_completion_stream — delegate to chat_completion
# ---------------------------------------------------------------------------


async def test_fake_stream_returns_openai_response() -> None:
    client = FakeChatAPIClient()
    result = await client.chat_completion_stream([_user("Hello")])
    assert isinstance(result, OpenAIResponse)


async def test_fake_stream_calls_on_delta_with_full_content() -> None:
    client = FakeChatAPIClient()
    received: list[str] = []

    async def on_delta(chunk: str) -> None:
        received.append(chunk)

    result = await client.chat_completion_stream([_user("test")], on_delta=on_delta)
    content = result.choices[0].message.content or ""
    # on_delta should have been called with the full content in one shot
    assert received == [content]


async def test_fake_stream_no_on_delta_does_not_raise() -> None:
    client = FakeChatAPIClient()
    result = await client.chat_completion_stream([_user("silent")])
    assert isinstance(result, OpenAIResponse)


async def test_fake_stream_refinement_marker_triggers_json() -> None:
    """Refinement marker still works via the streaming path."""
    client = FakeChatAPIClient()
    messages = [
        _system("You are a prompt-refinement assistant."),
        _user("Refine this."),
    ]
    received: list[str] = []

    async def on_delta(chunk: str) -> None:
        received.append(chunk)

    result = await client.chat_completion_stream(messages, on_delta=on_delta)
    content = result.choices[0].message.content or ""
    assert "proposed_delta" in content
    assert received == [content]


async def test_fake_stream_no_on_delta_when_empty_content() -> None:
    """No crash when content is empty (edge-case guard in override)."""

    class _EmptyFake(FakeChatAPIClient):
        async def chat_completion(
            self,
            messages: list[ChatMessage],
            max_tokens: int = 1024,
            temperature: float = 0.7,
        ) -> OpenAIResponse:
            from companion_bot_core.dev.fake_client import _make_openai_response  # noqa: PLC0415

            return _make_openai_response("")

    client = _EmptyFake()
    received: list[str] = []

    async def on_delta(chunk: str) -> None:
        received.append(chunk)

    await client.chat_completion_stream([_user("hi")], on_delta=on_delta)
    # Empty content must not trigger on_delta
    assert received == []
