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

import logging
from typing import Any, cast

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from tdbot.inference.circuit_breaker import CircuitBreaker
from tdbot.inference.schemas import ChatMessage, OpenAIResponse

log = logging.getLogger(__name__)

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
        model:           Model identifier (e.g. ``"gpt-4o-mini"``).
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
        response = await self._http.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
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

        raw = cast(
            "dict[str, Any]",
            await self._circuit_breaker.call(
                self._raw_completion,
                payload,
                max_tokens,
                temperature,
            ),
        )
        return OpenAIResponse.model_validate(raw)
