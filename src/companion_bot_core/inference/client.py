"""Async HTTP client for the OpenAI Chat Completions API.

Features
--------
- Retry with exponential back-off and jitter (via tenacity) for transient errors:
  HTTP 429, 500, 502, 503, 504, and network-level timeouts / connection errors.
- Non-retryable errors (HTTP 400, 401, 403, …) are raised immediately.
- Circuit breaker that opens after ``failure_threshold`` exhausted-retry calls,
  preventing further traffic when the provider is persistently unavailable.
- Streaming mode: ``chat_completion_stream`` calls the endpoint with
  ``stream=True`` and invokes an ``on_delta`` callback for each content chunk.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from companion_bot_core.inference.circuit_breaker import CircuitBreaker
from companion_bot_core.inference.schemas import ChatMessage, OpenAIResponse
from companion_bot_core.logging_config import get_logger

log = get_logger(__name__)

_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    """Return True for exceptions that warrant a retry attempt."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


async def _noop_delta(_chunk: str) -> None:
    """No-op delta callback used when the caller does not need per-chunk processing."""


class ChatAPIClient:
    """Async OpenAI Chat Completions client with retry and circuit breaking.

    Args:
        api_key:         Provider API key (sent as ``Authorization: Bearer …``).
        model:           Model identifier (e.g. ``"gpt-5-mini"``).
        base_url:        API base URL (default ``https://api.openai.com/v1``).
        timeout:         Per-request timeout in seconds (default 30.0).
        circuit_breaker: Custom CircuitBreaker instance; a default one is
                         created if not provided.
        http_client:     Injected httpx.AsyncClient (useful for testing).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        circuit_breaker: CircuitBreaker | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._is_gpt5_family = model.startswith("gpt-5")
        self._max_tokens_param = (
            "max_completion_tokens" if self._is_gpt5_family else "max_tokens"
        )
        self._http = http_client or httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()

    async def __aenter__(self) -> ChatAPIClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1.0, max=16.0),
        reraise=True,
    )
    async def _raw_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Make one HTTP request to the Chat Completions endpoint.

        Decorated with tenacity retry; this method should only be called via
        ``chat_completion`` which wraps it in the circuit breaker.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            self._max_tokens_param: max_tokens,
        }
        if not self._is_gpt5_family:
            payload["temperature"] = temperature
        response = await self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        return cast("dict[str, Any]", response.json())

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> OpenAIResponse:
        """Call the Chat Completions API and return a validated response.

        Args:
            messages:    Full message list (system + history + new user turn).
            max_tokens:  Maximum completion tokens (default 1024).
            temperature: Sampling temperature (default 0.7).

        Returns:
            Validated ``OpenAIResponse`` instance.

        Raises:
            CircuitBreakerOpen: Provider error rate exceeded the threshold.
            httpx.HTTPStatusError: Non-retryable HTTP error (400, 401, …).
            tenacity.RetryError: All retry attempts exhausted.
        """
        payload = [{"role": m.role, "content": m.content} for m in messages]

        call_start = time.perf_counter()
        raw = cast(
            "dict[str, Any]",
            await self._circuit_breaker.call(
                self._raw_completion,
                payload,
                max_tokens,
                temperature,
            ),
        )
        elapsed_ms = round((time.perf_counter() - call_start) * 1000, 2)
        response = OpenAIResponse.model_validate(raw)
        log.info(
            "model_api_call_completed",
            model=self._model,
            elapsed_ms=elapsed_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        return response


    async def _raw_completion_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        on_delta: Callable[[str], Awaitable[None]],
    ) -> dict[str, Any]:
        """Make a single streaming HTTP request to the Chat Completions endpoint.

        Parses Server-Sent Events (SSE) and calls *on_delta* for each text
        chunk.  Returns a raw dict that can be validated as an
        :class:`~companion_bot_core.inference.schemas.OpenAIResponse`.

        Unlike ``_raw_completion``, this method does **not** carry a retry
        decorator because retrying mid-stream would replay already-delivered
        chunks through *on_delta*.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            self._max_tokens_param: max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if not self._is_gpt5_family:
            payload["temperature"] = temperature

        full_content = ""
        full_refusal: str | None = None
        finish_reason: str | None = None
        response_id = ""
        usage_dict: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        async with self._http.stream("POST", "/chat/completions", json=payload) as http_resp:
            http_resp.raise_for_status()
            async for line in http_resp.aiter_lines():
                # Normalise any trailing whitespace / carriage-return from SSE framing
                # before all subsequent checks so the prefix and [DONE] tests are
                # consistent.
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if not response_id and "id" in data:
                    response_id = str(data["id"])

                # Usage arrives in a terminal chunk that has an empty choices list.
                raw_usage = data.get("usage")
                if isinstance(raw_usage, dict) and raw_usage:
                    usage_dict = {
                        "prompt_tokens": int(raw_usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
                        "total_tokens": int(raw_usage.get("total_tokens", 0)),
                    }

                choices: list[Any] = data.get("choices") or []
                if choices:
                    choice = choices[0]
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = str(fr)
                    delta: dict[str, Any] = choice.get("delta") or {}
                    delta_content = delta.get("content") or ""
                    if delta_content:
                        full_content += delta_content
                        await on_delta(delta_content)
                    delta_refusal = delta.get("refusal")
                    if delta_refusal:
                        full_refusal = (full_refusal or "") + str(delta_refusal)

        return {
            "id": response_id or "stream-unknown",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                        "refusal": full_refusal,
                    },
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": usage_dict,
        }

    async def chat_completion_stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> OpenAIResponse:
        """Call the Chat Completions API in streaming mode.

        Invokes *on_delta* with each text chunk as it is received so the
        caller can forward tokens to the end-user before the full response is
        complete.  When *on_delta* is ``None`` the chunks are accumulated
        silently and only the final :class:`OpenAIResponse` is returned.

        Unlike :meth:`chat_completion`, this method does **not** retry
        mid-stream — retrying would re-deliver already-streamed chunks to the
        user.  The circuit breaker is still applied: if it is open the call
        raises :class:`~companion_bot_core.inference.circuit_breaker.CircuitBreakerOpen`
        before any streaming begins.

        Args:
            messages:    Full message list (system + history + new user turn).
            max_tokens:  Maximum completion tokens (default 1024).
            temperature: Sampling temperature (default 0.7).
            on_delta:    Async callback invoked with each content chunk.
                         ``None`` accumulates chunks without callbacks.

        Returns:
            Validated ``OpenAIResponse`` with the fully accumulated reply.

        Raises:
            CircuitBreakerOpen: Provider error rate exceeded the threshold.
            httpx.HTTPStatusError: Non-retryable HTTP error (400, 401, …).
        """
        delta_fn: Callable[[str], Awaitable[None]] = (
            on_delta if on_delta is not None else _noop_delta
        )
        payload = [{"role": m.role, "content": m.content} for m in messages]

        call_start = time.perf_counter()
        raw = cast(
            "dict[str, Any]",
            await self._circuit_breaker.call(
                self._raw_completion_stream,
                payload,
                max_tokens,
                temperature,
                delta_fn,
            ),
        )
        elapsed_ms = round((time.perf_counter() - call_start) * 1000, 2)
        response = OpenAIResponse.model_validate(raw)
        log.info(
            "model_api_stream_completed",
            model=self._model,
            elapsed_ms=elapsed_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        return response
