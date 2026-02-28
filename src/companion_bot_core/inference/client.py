"""Async HTTP client for the OpenAI Chat Completions API.

Features
--------
- Retry with exponential back-off and jitter (via tenacity) for transient errors:
  HTTP 429, 500, 502, 503, 504, and network-level timeouts / connection errors.
- Non-retryable errors (HTTP 400, 401, 403, …) are raised immediately.
- Circuit breaker that opens after ``failure_threshold`` exhausted-retry calls,
  preventing further traffic when the provider is persistently unavailable.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, cast

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from companion_bot_core.inference.circuit_breaker import CircuitBreaker
from companion_bot_core.inference.schemas import (
    ChatMessage,
    OpenAIResponse,
    _OpenAIStreamChunk,
    _StreamEnd,
)
from companion_bot_core.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

log = get_logger(__name__)

_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    """Return True for exceptions that warrant a retry attempt."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


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

    async def chat_completion_stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str | _StreamEnd, None]:
        """Stream the Chat Completions API response as text chunks.

        Yields individual text tokens as they arrive via Server-Sent Events,
        then yields a single :class:`_StreamEnd` sentinel carrying final token
        usage, the finish reason, and whether the model issued a refusal.
        The sentinel is emitted only when the provider supports
        ``stream_options.include_usage``; callers must tolerate its absence.

        Unlike :meth:`chat_completion`, this path does not retry on failure.
        The circuit breaker still applies: if the circuit is open a
        ``CircuitBreakerOpen`` exception is raised before any yielding begins.

        Args:
            messages:    Full message list (system + history + new user turn).
            max_tokens:  Maximum completion tokens (default 1024).
            temperature: Sampling temperature (default 0.7; ignored for gpt-5
                         family models).

        Yields:
            ``str`` — incremental text token from the model.
            ``_StreamEnd`` — final sentinel with usage stats and refusal flag (last item).

        Raises:
            CircuitBreakerOpen: Provider error rate exceeded the threshold.
            httpx.HTTPStatusError: Non-retryable HTTP error.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            self._max_tokens_param: max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if not self._is_gpt5_family:
            payload["temperature"] = temperature

        call_start = time.perf_counter()
        await self._circuit_breaker._check_state()
        try:
            finish_reason = "stop"
            refusal_seen = False
            async with self._http.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        raw = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    try:
                        chunk = _OpenAIStreamChunk.model_validate(raw)
                    except Exception as exc:  # noqa: BLE001
                        log.debug(
                            "stream_chunk_validation_failed",
                            model=self._model,
                            error=str(exc),
                        )
                        continue
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if choice.delta.content:
                            yield choice.delta.content
                        if choice.delta.refusal:
                            refusal_seen = True
                        if choice.finish_reason is not None:
                            finish_reason = choice.finish_reason
                    if chunk.usage is not None:
                        yield _StreamEnd(
                            finish_reason=finish_reason,
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                            refusal=refusal_seen,
                        )
        except Exception:
            await self._circuit_breaker._record_failure()
            raise
        elapsed_ms = round((time.perf_counter() - call_start) * 1000, 2)
        await self._circuit_breaker._record_success()
        log.info(
            "model_api_stream_completed",
            model=self._model,
            elapsed_ms=elapsed_ms,
        )
