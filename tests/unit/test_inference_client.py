"""Unit tests for tdbot.inference.client (ChatAPIClient)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from tdbot.inference.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from tdbot.inference.client import ChatAPIClient, _is_retryable
from tdbot.inference.schemas import ChatMessage, OpenAIResponse

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
) -> tuple[ChatAPIClient, AsyncMock]:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    if http_response is not None:
        mock_http.post = AsyncMock(return_value=http_response)
    mock_http.aclose = AsyncMock()
    cb = circuit_breaker or CircuitBreaker(failure_threshold=10)
    client = ChatAPIClient(
        api_key="sk-test",
        model="gpt-4o-mini",
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
    client, mock_http = _make_client(http_response=_make_http_response())
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
