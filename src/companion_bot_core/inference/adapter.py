"""High-level chat inference adapter.

Provides a single public function, ``generate_reply``, that assembles the
full message list from per-user context, calls the model provider, validates
the response, and returns a typed ``InferenceReply``.
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
    on_delta: Callable[[str], Awaitable[None]],
) -> InferenceReply:
    """Generate a companion reply in streaming mode.

    Assembles the same message list as :func:`generate_reply` and calls
    ``client.chat_completion_stream``, forwarding each content chunk to
    *on_delta* as it arrives.  Returns the full :class:`InferenceReply` once
    the stream is exhausted.

    Args:
        client:       Configured ``ChatAPIClient`` instance.
        user_context: Per-user context including compiled system prompt and
                      recent conversation window.
        message:      The new incoming user message.
        on_delta:     Async callback invoked with each text chunk.

    Returns:
        ``InferenceReply`` containing the full accumulated reply, token usage,
        and safety metadata.

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
        max_tokens=user_context.max_tokens,
        on_delta=on_delta,
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
