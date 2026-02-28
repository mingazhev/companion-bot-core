"""Chat inference package: model adapter, retry client, and circuit breaker."""

from __future__ import annotations

from companion_bot_core.inference.adapter import generate_reply, generate_reply_stream
from companion_bot_core.inference.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from companion_bot_core.inference.client import ChatAPIClient
from companion_bot_core.inference.schemas import (
    ChatMessage,
    InferenceReply,
    SafetyFlags,
    TokenUsage,
    UserContext,
)

__all__ = [
    "ChatAPIClient",
    "ChatMessage",
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "InferenceReply",
    "SafetyFlags",
    "TokenUsage",
    "UserContext",
    "generate_reply",
    "generate_reply_stream",
]
