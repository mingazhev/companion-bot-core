"""Unit tests for ChatAPIClient streaming (chat_completion_stream)."""

from __future__ import annotations

import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from companion_bot_core.inference.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from companion_bot_core.inference.client import ChatAPIClient, _noop_delta
from companion_bot_core.inference.schemas import ChatMessage, OpenAIResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MESSAGES = [
    ChatMessage(role="system", content="You are helpful."),
    ChatMessage(role="user", content="Hi"),
]


def _sse_lines(*chunks: str, finish_reason: str = "stop") -> list[str]:
    """Build a list of SSE data lines from content chunks plus a usage line."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks):
        fr = finish_reason if i == len(chunks) - 1 else None
        data: dict[str, Any] = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": fr,
                }
            ],
            "usage": None,
        }
        lines.append("data: " + json.dumps(data))
    # Final usage-only event (stream_options: include_usage)
    lines.append(
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": len(chunks),
                    "total_tokens": 10 + len(chunks),
                },
            }
        )
    )
    lines.append("data: [DONE]")
    return lines


def _make_streaming_client(
    sse_lines: list[str],
    model: str = "gpt-4o-mini",
    circuit_breaker: CircuitBreaker | None = None,
) -> ChatAPIClient:
    """Build a ChatAPIClient backed by a mock that replays *sse_lines*."""

    async def _aiter_lines() -> Any:
        for line in sse_lines:
            yield line

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = _aiter_lines

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.aclose = AsyncMock()

    @contextlib.asynccontextmanager  # type: ignore[arg-type]
    async def _mock_stream(*_args: Any, **_kwargs: Any) -> Any:
        yield mock_response

    mock_http.stream = _mock_stream

    cb = circuit_breaker or CircuitBreaker(failure_threshold=10)
    return ChatAPIClient(
        api_key="sk-test",
        model=model,
        circuit_breaker=cb,
        http_client=mock_http,
    )


# ---------------------------------------------------------------------------
# _noop_delta
# ---------------------------------------------------------------------------


async def test_noop_delta_is_awaitable() -> None:
    # Should not raise; returns None
    result = await _noop_delta("hello")
    assert result is None


# ---------------------------------------------------------------------------
# chat_completion_stream — basic correctness
# ---------------------------------------------------------------------------


async def test_stream_returns_openai_response() -> None:
    client = _make_streaming_client(_sse_lines("Hello", " World"))
    result = await client.chat_completion_stream(_MESSAGES)
    assert isinstance(result, OpenAIResponse)


async def test_stream_accumulates_full_content() -> None:
    client = _make_streaming_client(_sse_lines("Hello", " World", "!"))
    result = await client.chat_completion_stream(_MESSAGES)
    assert result.choices[0].message.content == "Hello World!"


async def test_stream_calls_on_delta_per_chunk() -> None:
    received: list[str] = []

    async def on_delta(chunk: str) -> None:
        received.append(chunk)

    client = _make_streaming_client(_sse_lines("Hello", " there"))
    await client.chat_completion_stream(_MESSAGES, on_delta=on_delta)

    assert received == ["Hello", " there"]


async def test_stream_on_delta_none_does_not_raise() -> None:
    client = _make_streaming_client(_sse_lines("Silent", " stream"))
    result = await client.chat_completion_stream(_MESSAGES, on_delta=None)
    assert result.choices[0].message.content == "Silent stream"


async def test_stream_records_usage() -> None:
    client = _make_streaming_client(_sse_lines("Hi"))
    result = await client.chat_completion_stream(_MESSAGES)
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 1
    assert result.usage.total_tokens == 11


async def test_stream_finish_reason_preserved() -> None:
    client = _make_streaming_client(_sse_lines("Bye"), model="gpt-4o-mini")
    result = await client.chat_completion_stream(_MESSAGES)
    assert result.choices[0].finish_reason == "stop"


async def test_stream_content_filter_finish_reason() -> None:
    client = _make_streaming_client(_sse_lines("...", finish_reason="content_filter"))
    result = await client.chat_completion_stream(_MESSAGES)
    assert result.choices[0].finish_reason == "content_filter"


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


async def test_stream_sends_stream_true_in_payload() -> None:
    captured_kwargs: dict[str, Any] = {}

    async def _aiter_lines() -> Any:
        for line in _sse_lines("OK"):
            yield line

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = _aiter_lines
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.aclose = AsyncMock()

    @contextlib.asynccontextmanager  # type: ignore[arg-type]
    async def _mock_stream(*_args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        yield mock_response

    mock_http.stream = _mock_stream

    cb = CircuitBreaker(failure_threshold=10)
    client = ChatAPIClient(
        api_key="sk-test", model="gpt-4o-mini", circuit_breaker=cb, http_client=mock_http
    )
    await client.chat_completion_stream(_MESSAGES, max_tokens=256, temperature=0.5)

    body = captured_kwargs.get("json", {})
    assert body["stream"] is True
    assert body["model"] == "gpt-4o-mini"
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0.5


async def test_stream_gpt5_uses_max_completion_tokens() -> None:
    captured_kwargs: dict[str, Any] = {}

    async def _aiter_lines() -> Any:
        for line in _sse_lines("OK"):
            yield line

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = _aiter_lines
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.aclose = AsyncMock()

    @contextlib.asynccontextmanager  # type: ignore[arg-type]
    async def _mock_stream(*_args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        yield mock_response

    mock_http.stream = _mock_stream

    cb = CircuitBreaker(failure_threshold=10)
    client = ChatAPIClient(
        api_key="sk-test", model="gpt-5-mini", circuit_breaker=cb, http_client=mock_http
    )
    await client.chat_completion_stream(_MESSAGES, max_tokens=512)

    body = captured_kwargs.get("json", {})
    assert body.get("max_completion_tokens") == 512
    assert "max_tokens" not in body
    assert "temperature" not in body


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------


async def test_stream_raises_circuit_breaker_open() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999.0)
    # Make the first call fail so the circuit trips
    async def _aiter_fail() -> Any:
        # Declare as an async generator by having a yield; the raise fires first.
        if False:  # pragma: no cover
            yield
        raise httpx.NetworkError("connection refused")

    mock_response_fail = AsyncMock()
    mock_response_fail.raise_for_status = MagicMock()
    mock_response_fail.aiter_lines = _aiter_fail
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.aclose = AsyncMock()

    @contextlib.asynccontextmanager  # type: ignore[arg-type]
    async def _mock_stream_fail(*_args: Any, **_kwargs: Any) -> Any:
        yield mock_response_fail

    mock_http.stream = _mock_stream_fail

    client = ChatAPIClient(
        api_key="sk-test", model="gpt-4o-mini", circuit_breaker=cb, http_client=mock_http
    )

    with pytest.raises(httpx.NetworkError):
        await client.chat_completion_stream(_MESSAGES)

    assert cb.state.value == "open"

    with pytest.raises(CircuitBreakerOpen):
        await client.chat_completion_stream(_MESSAGES)


# ---------------------------------------------------------------------------
# Refusal tracking
# ---------------------------------------------------------------------------


async def test_stream_tracks_refusal_in_response() -> None:
    """Delta chunks containing 'refusal' are captured in message.refusal."""

    async def _aiter_lines() -> Any:
        yield "data: " + json.dumps(
            {
                "id": "x",
                "choices": [
                    {"index": 0, "delta": {"refusal": "I can't help."}, "finish_reason": "stop"}
                ],
                "usage": None,
            }
        )
        yield "data: " + json.dumps(
            {
                "id": "x",
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
            }
        )
        yield "data: [DONE]"

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = _aiter_lines
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.aclose = AsyncMock()

    @contextlib.asynccontextmanager  # type: ignore[arg-type]
    async def _mock_stream(*_a: Any, **_kw: Any) -> Any:
        yield mock_response

    mock_http.stream = _mock_stream
    client = ChatAPIClient(
        api_key="sk-test",
        model="gpt-4o-mini",
        circuit_breaker=CircuitBreaker(failure_threshold=10),
        http_client=mock_http,
    )
    result = await client.chat_completion_stream(_MESSAGES)
    assert result.choices[0].message.refusal == "I can't help."
    assert (result.choices[0].message.content or "") == ""
