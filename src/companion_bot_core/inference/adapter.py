"""High-level chat inference adapter.

Provides two public functions:

- ``generate_reply``: assembles the full message list from per-user context,
  calls the model provider (blocking), validates the response, and returns a
  typed ``InferenceReply``.
- ``generate_reply_stream``: same assembly, but calls the provider using
  Server-Sent Events streaming.  Each arriving text token is forwarded to a
  caller-supplied ``on_chunk`` callback so that a Telegram handler can
  progressively edit a sent message.  Returns the same ``InferenceReply``
  type after the stream ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from companion_bot_core.inference.schemas import (
    ChatMessage,
    InferenceReply,
    SafetyFlags,
    TokenUsage,
    UserContext,
    _StreamEnd,
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
    on_chunk: Callable[[str], Awaitable[None]],
) -> InferenceReply:
    """Generate a companion reply using streaming, forwarding chunks to *on_chunk*.

    Message assembly order is identical to :func:`generate_reply`:
        1. System prompt (from ``user_context.system_prompt``)
        2. Conversation history (from ``user_context.conversation_history``)
        3. New user turn (``message``)

    Each text token received from the provider is forwarded to *on_chunk*
    immediately, allowing the caller (e.g. a Telegram handler) to progressively
    update a sent message before the full reply is available.

    Args:
        client:       Configured ``ChatAPIClient`` instance.
        user_context: Per-user context including compiled system prompt and
                      recent conversation window.
        message:      The new incoming user message.
        on_chunk:     Async callback invoked with each text token chunk as it
                      arrives.  Exceptions raised by the callback are propagated
                      to the caller.

    Returns:
        ``InferenceReply`` containing the full accumulated reply text, token
        usage (from the streaming usage sentinel when available, otherwise
        zeroed), and safety metadata.

    Raises:
        CircuitBreakerOpen: When the model provider is temporarily unavailable.
        httpx.HTTPStatusError: On non-retryable API errors (400, 401, …).
    """
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=user_context.system_prompt),
        *user_context.conversation_history,
        ChatMessage(role="user", content=message),
    ]

    accumulated = ""
    end: _StreamEnd | None = None

    async for item in client.chat_completion_stream(
        messages,
        max_tokens=user_context.max_tokens,
    ):
        if isinstance(item, _StreamEnd):
            end = item
        else:
            accumulated += item
            await on_chunk(item)

    finish_reason = end.finish_reason if end is not None else "stop"
    return InferenceReply(
        reply=accumulated,
        usage=TokenUsage(
            prompt_tokens=end.prompt_tokens if end is not None else 0,
            completion_tokens=end.completion_tokens if end is not None else 0,
            total_tokens=end.total_tokens if end is not None else 0,
        ),
        safety_flags=SafetyFlags(
            content_filtered=finish_reason == "content_filter",
            refusal=end.refusal if end is not None else False,
            finish_reason=finish_reason,
        ),
    )
