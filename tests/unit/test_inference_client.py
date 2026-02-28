"""Unit tests for companion_bot_core.inference.client (ChatAPIClient)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from companion_bot_core.inference.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from companion_bot_core.inference.client import ChatAPIClient, _is_retryable
from companion_bot_core.inference.schemas import ChatMessage, OpenAIResponse, _StreamEnd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RESPONSE_BODY = {
    "id": "chatcmpl-test",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!", "refusal": None},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

_MESSAGES = [
    ChatMessage(role="system", content="You are helpful."),
    ChatMessage(role="user", content="Hi"),
]


def _make_http_response(status_code: int = 200, body: object = None) -> httpx.Response:
    content = json.dumps(body or _VALID_RESPONSE_BODY).encode()
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(status_code=status_code, content=content, request=request)


def _make_client(
    http_response: httpx.Response | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    model: str = "gpt-4o-mini",
) -> tuple[ChatAPIClient, AsyncMock]:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    if http_response is not None:
        mock_http.post = AsyncMock(return_value=http_response)
    mock_http.aclose = AsyncMock()
    cb = circuit_breaker or CircuitBreaker(failure_threshold=10)
    client = ChatAPIClient(
        api_key="sk-test",
        model=model,
        circuit_breaker=cb,
        http_client=mock_http,
    )
    return client, mock_http


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------


def test_is_retryable_timeout() -> None:
    assert _is_retryable(httpx.TimeoutException("timeout"))


def test_is_retryable_network_error() -> None:
    assert _is_retryable(httpx.NetworkError("conn refused"))


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_is_retryable_retryable_status(status: int) -> None:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(status_code=status, request=req)
    exc = httpx.HTTPStatusError(f"{status}", request=req, response=resp)
    assert _is_retryable(exc)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_is_not_retryable_client_error(status: int) -> None:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(status_code=status, request=req)
    exc = httpx.HTTPStatusError(f"{status}", request=req, response=resp)
    assert not _is_retryable(exc)


def test_is_not_retryable_generic_exception() -> None:
    assert not _is_retryable(ValueError("nope"))


# ---------------------------------------------------------------------------
# Successful chat_completion
# ---------------------------------------------------------------------------


async def test_chat_completion_returns_validated_response() -> None:
    client, _ = _make_client(http_response=_make_http_response())
    result = await client.chat_completion(_MESSAGES)
    assert isinstance(result, OpenAIResponse)
    assert result.choices[0].message.content == "Hello!"
    assert result.usage.total_tokens == 15


async def test_chat_completion_sends_correct_payload() -> None:
    client, mock_http = _make_client(http_response=_make_http_response(), model="gpt-4o-mini")
    await client.chat_completion(_MESSAGES, max_tokens=512, temperature=0.5)

    mock_http.post.assert_called_once()
    _, kwargs = mock_http.post.call_args
    body = kwargs["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["max_tokens"] == 512
    assert body["temperature"] == 0.5
    assert body["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]


async def test_chat_completion_uses_max_completion_tokens_for_gpt5_models() -> None:
    client, mock_http = _make_client(http_response=_make_http_response(), model="gpt-5-mini")
    await client.chat_completion(_MESSAGES, max_tokens=256, temperature=0.2)

    mock_http.post.assert_called_once()
    _, kwargs = mock_http.post.call_args
    body = kwargs["json"]
    assert body["model"] == "gpt-5-mini"
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body
    assert "temperature" not in body


# ---------------------------------------------------------------------------
# Non-retryable HTTP error propagates immediately
# ---------------------------------------------------------------------------


async def test_chat_completion_raises_on_401() -> None:
    client, mock_http = _make_client()
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    mock_http.post = AsyncMock(
        side_effect=httpx.HTTPStatusError("401", request=req, response=resp)
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.chat_completion(_MESSAGES)

    assert exc_info.value.response.status_code == 401


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------


async def test_chat_completion_raises_when_circuit_open() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999.0)
    client, mock_http = _make_client(circuit_breaker=cb)

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(500, request=req)
    mock_http.post = AsyncMock(
        side_effect=httpx.HTTPStatusError("500", request=req, response=resp)
    )

    # Exhaust retries to open the circuit (3 attempts per call × 1 threshold)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completion(_MESSAGES)

    assert cb.state.value == "open"

    with pytest.raises(CircuitBreakerOpen):
        await client.chat_completion(_MESSAGES)


# ---------------------------------------------------------------------------
# Context manager usage
# ---------------------------------------------------------------------------


async def test_client_context_manager_closes_http() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.aclose = AsyncMock()
    client = ChatAPIClient(
        api_key="sk-test",
        model="gpt-4o-mini",
        http_client=mock_http,
    )
    async with client:
        pass
    mock_http.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# Streaming: chat_completion_stream
# ---------------------------------------------------------------------------


def _make_sse_lines(chunks: list[str], finish_reason: str = "stop") -> list[str]:
    """Build a minimal SSE line sequence for the given text chunks."""
    lines: list[str] = []
    for chunk in chunks:
        body = {
            "id": "chatcmpl-stream",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        lines.append(f"data: {json.dumps(body)}")
    # Final choice chunk with finish_reason
    final = {
        "id": "chatcmpl-stream",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    lines.append(f"data: {json.dumps(final)}")
    # Usage sentinel
    usage = {
        "id": "chatcmpl-stream",
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(chunks),
            "total_tokens": 10 + len(chunks),
        },
    }
    lines.append(f"data: {json.dumps(usage)}")
    lines.append("data: [DONE]")
    return lines


def _make_streaming_http(lines: list[str]) -> AsyncMock:
    """Return an httpx.AsyncClient mock that streams the given SSE lines."""
    mock_http = MagicMock()

    async def _aiter_lines():
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = _aiter_lines

    class _StreamCtx:
        async def __aenter__(self):
            return mock_response

        async def __aexit__(self, *args):
            pass

    mock_http.stream = MagicMock(return_value=_StreamCtx())
    mock_http.aclose = AsyncMock()
    return mock_http


async def test_chat_completion_stream_yields_text_chunks() -> None:
    mock_http = _make_streaming_http(_make_sse_lines(["Hello", " world"]))
    cb = CircuitBreaker(failure_threshold=10)
    client = ChatAPIClient(
        api_key="sk-test", model="gpt-4o-mini", circuit_breaker=cb,
        http_client=mock_http,
    )

    collected = []
    async for item in client.chat_completion_stream(_MESSAGES):
        if isinstance(item, str):
            collected.append(item)

    assert collected == ["Hello", " world"]


async def test_chat_completion_stream_yields_stream_end_sentinel() -> None:
    mock_http = _make_streaming_http(_make_sse_lines(["Hi"], finish_reason="stop"))
    cb = CircuitBreaker(failure_threshold=10)
    client = ChatAPIClient(
        api_key="sk-test", model="gpt-4o-mini", circuit_breaker=cb,
        http_client=mock_http,
    )

    sentinel = None
    async for item in client.chat_completion_stream(_MESSAGES):
        if isinstance(item, _StreamEnd):
            sentinel = item

    assert sentinel is not None
    assert sentinel.finish_reason == "stop"
    assert sentinel.prompt_tokens == 10
    assert sentinel.completion_tokens == 1


async def test_chat_completion_stream_sends_correct_payload() -> None:
    mock_http = _make_streaming_http(_make_sse_lines(["ok"]))
    cb = CircuitBreaker(failure_threshold=10)
    client = ChatAPIClient(
        api_key="sk-test", model="gpt-4o-mini", circuit_breaker=cb,
        http_client=mock_http,
    )

    async for _ in client.chat_completion_stream(_MESSAGES, max_tokens=256, temperature=0.3):
        pass

    mock_http.stream.assert_called_once()
    call_kwargs = mock_http.stream.call_args[1]
    body = call_kwargs["json"]
    assert body["stream"] is True
    assert body["model"] == "gpt-4o-mini"
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0.3
    assert body.get("stream_options", {}).get("include_usage") is True


async def test_chat_completion_stream_raises_when_circuit_open() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999.0)
    mock_http = _make_streaming_http([])
    client = ChatAPIClient(
        api_key="sk-test", model="gpt-4o-mini", circuit_breaker=cb,
        http_client=mock_http,
    )

    # Force circuit open
    from companion_bot_core.inference.circuit_breaker import CircuitState
    cb._state = CircuitState.OPEN
    cb._opened_at = 0.0
    cb._failure_count = 5

    with pytest.raises(CircuitBreakerOpen):
        async for _ in client.chat_completion_stream(_MESSAGES):
            pass


async def test_chat_completion_stream_records_circuit_failure_on_http_error() -> None:
    mock_http = MagicMock()

    class _ErrorStreamCtx:
        async def __aenter__(self):
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            resp = httpx.Response(500, request=req)
            raise httpx.HTTPStatusError("500", request=req, response=resp)

        async def __aexit__(self, *args):
            pass

    mock_http.stream = MagicMock(return_value=_ErrorStreamCtx())
    mock_http.aclose = AsyncMock()

    cb = CircuitBreaker(failure_threshold=10)
    client = ChatAPIClient(
        api_key="sk-test", model="gpt-4o-mini", circuit_breaker=cb,
        http_client=mock_http,
    )

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in client.chat_completion_stream(_MESSAGES):
            pass

    assert cb.failure_count == 1


async def test_chat_completion_stream_skips_malformed_json_lines() -> None:
    lines = [
        "data: not-valid-json",
        'data: {"id":"x","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}',
        "data: [DONE]",
    ]
    mock_http = _make_streaming_http(lines)
    cb = CircuitBreaker(failure_threshold=10)
    client = ChatAPIClient(
        api_key="sk-test", model="gpt-4o-mini", circuit_breaker=cb,
        http_client=mock_http,
    )

    collected = []
    async for item in client.chat_completion_stream(_MESSAGES):
        if isinstance(item, str):
            collected.append(item)
    assert collected == ["ok"]
