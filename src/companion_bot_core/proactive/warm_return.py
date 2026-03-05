"""Warm return detection for proactive messaging.

When a user returns after an extended absence (>= 48 hours), a warm return
instruction is injected into the system prompt.  This is distinct from the
1-hour continuity hint in ``context_loader`` — the warm return carries a
stronger "welcome back" tone and is always active (does not require opt-in).

The function consumes the ``activity_gap`` already computed by
:func:`~companion_bot_core.orchestrator.context_loader.build_continuity_hint`
so there is no extra Redis read.
"""

from __future__ import annotations

from companion_bot_core.i18n import normalize_locale, tr
from companion_bot_core.logging_config import get_logger

log = get_logger(__name__)

# 48 hours in seconds
WARM_RETURN_GAP_SECONDS = 48 * 3600


def build_warm_return_hint(
    activity_gap_seconds: int,
    locale: str | None = None,
) -> str:
    """Return a warm return instruction if the user has been away >= 48 h.

    Args:
        activity_gap_seconds: Seconds since the user's last message, as
            computed by ``build_continuity_hint``.
        locale: User locale for i18n.

    Returns:
        An instruction string or empty string when no warm return is needed.
    """
    if activity_gap_seconds < WARM_RETURN_GAP_SECONDS:
        return ""

    resolved = normalize_locale(locale)
    return tr("proactive.warm_return_instruction", resolved)
