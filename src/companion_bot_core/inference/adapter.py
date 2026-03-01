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
    _StreamEnd,
)
from companion_bot_core.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from companion_bot_core.inference.client import ChatAPIClient

log = get_logger(__name__)


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
    """Generate a companion reply with streaming, forwarding tokens via *on_chunk*.

    Same message assembly as :func:`generate_reply`.  Each content delta is
    forwarded to *on_chunk* as it arrives.  Returns the same
    :class:`InferenceReply` once the stream is fully consumed.

    Args:
        client:       Configured ``ChatAPIClient`` instance.
        user_context: Per-user context including compiled system prompt and
                      recent conversation window.
        message:      The new incoming user message.
        on_chunk:     Async callback invoked with each content delta string.

    Returns:
        ``InferenceReply`` containing the full reply text, token usage, and
        safety metadata.
    """
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=user_context.system_prompt),
        *user_context.conversation_history,
        ChatMessage(role="user", content=message),
    ]

    collected: list[str] = []
    stream_end: _StreamEnd | None = None

    try:
        async for item in client.chat_completion_stream(
            messages, max_tokens=user_context.max_tokens,
        ):
            if isinstance(item, _StreamEnd):
                stream_end = item
            else:
                collected.append(item)
                await on_chunk(item)
    except Exception:
        # Stream interrupted (timeout, network error, etc.). Return
        # whatever was collected so the user sees a partial reply
        # rather than an error message replacing the streamed text.
        if collected:
            log.warning(
                "stream_interrupted_returning_partial",
                collected_chars=sum(len(c) for c in collected),
            )
        else:
            raise

    if stream_end is None:
        stream_end = _StreamEnd(
            finish_reason="stop",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            refusal=False,
        )

    return InferenceReply(
        reply="".join(collected),
        usage=TokenUsage(
            prompt_tokens=stream_end.prompt_tokens,
            completion_tokens=stream_end.completion_tokens,
            total_tokens=stream_end.total_tokens,
        ),
        safety_flags=SafetyFlags(
            content_filtered=stream_end.finish_reason == "content_filter",
            refusal=stream_end.refusal,
            finish_reason=stream_end.finish_reason,
        ),
    )
