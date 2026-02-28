"""High-level chat inference adapter.

Provides two public functions:
- ``generate_reply``: standard (non-streaming) call.
- ``generate_reply_stream``: SSE streaming call with an ``on_partial_reply``
  callback invoked for each text chunk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from companion_bot_core.inference.schemas import (
    ChatMessage,
    InferenceReply,
    SafetyFlags,
    TokenUsage,
    UserContext,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from companion_bot_core.inference.client import ChatAPIClient


async def generate_reply(
    client: ChatAPIClient,
    user_context: UserContext,
    message: str,
) -> InferenceReply:
    """Generate a companion reply for *message* given the user's context.

    Message assembly order:
        1. System prompt (from ``user_context.system_prompt``)
        2. Conversation history (from ``user_context.conversation_history``)
        3. New user turn (``message``)

    Args:
        client:       Configured ``ChatAPIClient`` instance.
        user_context: Per-user context including compiled system prompt and
                      recent conversation window.
        message:      The new incoming user message.

    Returns:
        ``InferenceReply`` containing the reply text, token usage, and safety
        metadata.

    Raises:
        CircuitBreakerOpen: When the model provider is temporarily unavailable.
        httpx.HTTPStatusError: On non-retryable API errors (400, 401, …).
        pydantic.ValidationError: When the provider response fails schema
                                  validation.
    """
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=user_context.system_prompt),
        *user_context.conversation_history,
        ChatMessage(role="user", content=message),
    ]

    response = await client.chat_completion(
        messages,
        max_tokens=user_context.max_tokens,
    )

    choice = response.choices[0]
    finish_reason = choice.finish_reason or "stop"

    return InferenceReply(
        reply=choice.message.content or "",
        usage=TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        ),
        safety_flags=SafetyFlags(
            content_filtered=finish_reason == "content_filter",
            refusal=choice.message.refusal is not None,
            finish_reason=finish_reason,
        ),
    )


async def generate_reply_stream(
    client: ChatAPIClient,
    user_context: UserContext,
    message: str,
    on_partial_reply: Callable[[str], Awaitable[None]],
) -> InferenceReply:
    """Generate a companion reply using SSE streaming.

    Identical to :func:`generate_reply` in terms of message assembly and
    return type, but invokes *on_partial_reply* with each text chunk as it
    arrives from the model provider so callers can forward incremental output
    to the user before the full reply is ready.

    Args:
        client:           Configured ``ChatAPIClient`` instance.
        user_context:     Per-user context including compiled system prompt and
                          recent conversation window.
        message:          The new incoming user message.
        on_partial_reply: Async callback invoked with each text delta.

    Returns:
        ``InferenceReply`` built from the fully accumulated reply text.

    Raises:
        CircuitBreakerOpen: When the model provider is temporarily unavailable.
        httpx.HTTPStatusError: On non-retryable API errors (400, 401, …).
        pydantic.ValidationError: When the provider response fails schema
                                  validation.
    """
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=user_context.system_prompt),
        *user_context.conversation_history,
        ChatMessage(role="user", content=message),
    ]

    response = await client.chat_completion_stream(
        messages,
        on_delta=on_partial_reply,
        max_tokens=user_context.max_tokens,
    )

    choice = response.choices[0]
    finish_reason = choice.finish_reason or "stop"

    return InferenceReply(
        reply=choice.message.content or "",
        usage=TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        ),
        safety_flags=SafetyFlags(
            content_filtered=finish_reason == "content_filter",
            refusal=choice.message.refusal is not None,
            finish_reason=finish_reason,
        ),
    )
