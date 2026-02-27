"""Fake ChatAPIClient for local development without real API credentials.

Returns deterministic, schema-valid responses so the full pipeline can be
exercised locally without sending traffic to any model provider.

Detects refinement calls by looking for the ``prompt-refinement assistant``
marker in the system prompt and returns appropriate JSON; all other calls
receive a conversational echo reply.

Usage in main.py::

    from tdbot.dev.fake_client import FakeChatAPIClient
    chat_client = FakeChatAPIClient()
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from tdbot.inference.client import ChatAPIClient
from tdbot.inference.schemas import ChatMessage, OpenAIResponse

# Substring that appears only in the refinement model's system prompt.
_REFINEMENT_MARKER: str = "prompt-refinement assistant"

# Canned JSON returned for refinement calls.  Returns a non-null
# ``long_term_profile`` delta so the worker exercises the full save path
# (new snapshot, audit event, user-visible notice) in dev mode.
_FAKE_REFINEMENT_JSON: str = json.dumps(
    {
        "proposed_delta": {
            "persona_segment": None,
            "skill_packs": None,
            "long_term_profile": "User enjoys general conversation topics.",
        },
        "rationale": "Simulated refinement — running in dev mode with fake adapter.",
        "risk_flags": [],
    }
)


def _make_openai_response(content: str) -> OpenAIResponse:
    """Build a minimal schema-valid :class:`~tdbot.inference.schemas.OpenAIResponse`."""
    return OpenAIResponse.model_validate(
        {
            "id": f"fake-{uuid.uuid4().hex[:8]}",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        }
    )


class FakeChatAPIClient(ChatAPIClient):
    """Drop-in replacement for :class:`~tdbot.inference.client.ChatAPIClient`.

    Never makes real HTTP requests.  Detects whether the call is a refinement
    request (via the system-prompt marker) and returns canned JSON; otherwise
    returns a simple echo reply prefixed with ``[Dev mode]``.

    Args:
        model: Label used in log messages (has no functional effect).
    """

    def __init__(self, model: str = "fake-model") -> None:
        # Skip super().__init__ to avoid creating a real httpx.AsyncClient
        # pointed at api.openai.com.  Set sentinel values for parent attributes
        # so that accidental access raises a clear error instead of AttributeError.
        self._model = model
        self._circuit_breaker = None  # type: ignore[assignment]
        self._http = None  # type: ignore[assignment]

    def _is_refinement_call(self, messages: list[ChatMessage]) -> bool:
        """Return True when the system prompt contains the refinement marker."""
        return any(
            _REFINEMENT_MARKER in m.content for m in messages if m.role == "system"
        )

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> OpenAIResponse:
        """Return a canned response without any network I/O.

        Args:
            messages:    Full message list assembled by the caller.
            max_tokens:  Ignored — included only for interface compatibility.
            temperature: Ignored — included only for interface compatibility.

        Returns:
            A schema-valid :class:`~tdbot.inference.schemas.OpenAIResponse`
            with either a JSON refinement delta or a plain echo reply.
        """
        _ = max_tokens, temperature  # unused in fake implementation
        if self._is_refinement_call(messages):
            return _make_openai_response(_FAKE_REFINEMENT_JSON)

        user_content = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "Hello!",
        )
        return _make_openai_response(f"[Dev mode] Echo: {user_content}")

    async def close(self) -> None:
        """No-op — no real HTTP client to close."""

    async def __aenter__(self) -> FakeChatAPIClient:
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Close on exit from async context."""
        await self.close()
