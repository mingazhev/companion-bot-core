"""Pydantic models for chat inference inputs, outputs, and OpenAI API responses.

Public surface:
    UserContext      — per-user context passed to generate_reply
    ChatMessage      — a single message in a conversation turn
    InferenceReply   — the result returned by generate_reply
    TokenUsage       — token consumption from the model provider
    SafetyFlags      — content filtering / refusal metadata

Internal (prefixed with _):
    _OpenAIMessage, _OpenAIChoice, _OpenAIUsage, OpenAIResponse
    — structural types for parsing the raw OpenAI Chat Completions response.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single turn in the conversation, with role and text content."""

    role: Literal["system", "user", "assistant"]
    content: str


class UserContext(BaseModel):
    """Per-user inference context assembled by the orchestrator before each call."""

    user_id: str = Field(description="Opaque user identifier for logging")
    system_prompt: str = Field(description="Compiled system prompt for this user")
    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Recent message window (oldest first, excluding the new turn)",
    )
    max_tokens: Annotated[int, Field(ge=1, le=4096)] = Field(
        default=1024,
        description="Upper bound on completion tokens",
    )


class TokenUsage(BaseModel):
    """Token counts reported by the model provider."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class SafetyFlags(BaseModel):
    """Content safety metadata extracted from the model response."""

    content_filtered: bool = Field(
        description="True when finish_reason is 'content_filter'"
    )
    refusal: bool = Field(
        description="True when the model explicitly refused the request"
    )
    finish_reason: str = Field(
        description="Raw finish_reason from the provider (stop, length, content_filter, …)"
    )


class InferenceReply(BaseModel):
    """Complete result returned by generate_reply."""

    reply: str
    usage: TokenUsage
    safety_flags: SafetyFlags


# ---------------------------------------------------------------------------
# Internal: OpenAI Chat Completions response shape
# ---------------------------------------------------------------------------


class _OpenAIMessage(BaseModel):
    role: str
    content: str | None = None
    refusal: str | None = None


class _OpenAIChoice(BaseModel):
    index: int = 0
    message: _OpenAIMessage
    finish_reason: str | None = None


class _OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponse(BaseModel):
    """Validated wrapper around the OpenAI Chat Completions API response body."""

    id: str
    choices: Annotated[list[_OpenAIChoice], Field(min_length=1)]
    usage: _OpenAIUsage


# ---------------------------------------------------------------------------
# Internal: OpenAI Chat Completions streaming response shape
# ---------------------------------------------------------------------------


class _OpenAIDelta(BaseModel):
    """Delta fragment in a streaming chunk (content may be absent on role-only events)."""

    content: str | None = None
    refusal: str | None = None


class _OpenAIStreamChoice(BaseModel):
    index: int = 0
    delta: _OpenAIDelta
    finish_reason: str | None = None


class _OpenAIStreamChunk(BaseModel):
    """One Server-Sent Event chunk from the streaming Chat Completions endpoint.

    ``choices`` is empty on the final usage-only chunk when
    ``stream_options: {"include_usage": true}`` is requested.
    ``usage`` is ``None`` on all intermediate text chunks.
    """

    id: str
    choices: list[_OpenAIStreamChoice] = Field(default_factory=list)
    usage: _OpenAIUsage | None = None


class _StreamEnd:
    """Sentinel object yielded last by :meth:`ChatAPIClient.chat_completion_stream`.

    Carries aggregated token-usage statistics and the terminal finish reason.
    Callers detect it with ``isinstance(item, _StreamEnd)``.
    """

    __slots__ = ("finish_reason", "prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(
        self,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
