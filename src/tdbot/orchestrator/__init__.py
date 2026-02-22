"""Conversation orchestrator package.

Assembles per-user context, routes behavior change intents, manages the
confirmation dialogue state, generates replies, and persists results.
"""

from __future__ import annotations

from tdbot.orchestrator.context_loader import load_recent_messages, load_user_context
from tdbot.orchestrator.dialogue_state import (
    PendingChange,
    clear_pending_change,
    get_pending_change,
    set_pending_change,
)
from tdbot.orchestrator.orchestrator import process_message

__all__ = [
    "PendingChange",
    "clear_pending_change",
    "get_pending_change",
    "load_recent_messages",
    "load_user_context",
    "process_message",
    "set_pending_change",
]
