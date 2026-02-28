"""structlog processor: redact PII fields before log records are rendered.

The processor replaces the *values* of known sensitive keys with the literal
string ``[REDACTED]``.  Keys that are not in the PII set pass through
unchanged, so operational metadata (correlation_id, user_id, etc.) remains
visible for debugging.

Usage
-----
Add ``redact_pii`` to the *shared_processors* list in :mod:`companion_bot_core.logging_config`
before any rendering step:

    shared_processors = [
        ...
        redact_pii,
        ...
        renderer,
    ]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import MutableMapping

_REDACTED = "[REDACTED]"

# Fields whose values may contain raw user content or sensitive persona text.
_PII_FIELDS: frozenset[str] = frozenset(
    {
        "content",        # conversation message content
        "message_text",   # raw incoming text
        "system_prompt",  # assembled system prompt (contains persona)
        "skill_prompts_json",  # map of skill → prompt text
        "style_constraints",  # free-text style constraints
        "reply",          # model-generated reply text
        "persona_name",   # user-chosen persona name
        "tone",           # user-chosen tone (encrypted at rest)
    }
)


def redact_pii(
    _logger: Any,
    _method: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Replace values of PII fields with ``[REDACTED]``.

    This is a structlog-compatible processor (accepts logger, method, event_dict;
    returns the modified event_dict).
    """
    for key in _PII_FIELDS:
        if key in event_dict:
            event_dict[key] = _REDACTED
    return event_dict
