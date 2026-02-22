"""Chat inference package: model adapter, retry client, and circuit breaker."""

from __future__ import annotations

from tdbot.inference.adapter import generate_reply
from tdbot.inference.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from tdbot.inference.client import ChatAPIClient
from tdbot.inference.schemas import (
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
]
