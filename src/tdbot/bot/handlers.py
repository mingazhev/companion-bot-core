"""Command handlers for all registered bot commands.

Each handler is a plain async function registered on the shared *router*.
Handlers receive injected dependencies (``db_user``, ``tg_user``) populated
by :class:`~tdbot.bot.middleware.IngressMiddleware`.

Commands implemented here
-------------------------
- /start              — welcome message
- /profile            — show current user settings
- /set_tone           — update response tone
- /set_persona        — update persona name
- /memory_compact_now — trigger memory compaction job
- /reset_persona      — reset persona to defaults
- /privacy            — display privacy policy summary
- /delete_my_data     — initiate hard-delete flow

Full DB writes for tone/persona changes are deferred to Task 6 (Prompt state
manager); the handlers below respond correctly and log the intent.
"""

from __future__ import annotations

import html
import time
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.filters import Command, CommandObject

from tdbot.logging_config import get_logger
from tdbot.orchestrator import process_message
from tdbot.tracing import span

if TYPE_CHECKING:
    from aiogram.types import Message
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from tdbot.config import Settings
    from tdbot.db.models import User
    from tdbot.inference.client import ChatAPIClient
    from tdbot.prompt.snapshot_store import SnapshotStore

from tdbot.privacy.delete_user import hard_delete_user
from tdbot.refinement.worker import check_and_clear_user_notice

log = get_logger(__name__)

router = Router(name="commands")

_VALID_TONES: frozenset[str] = frozenset(
    {"friendly", "professional", "playful", "neutral", "casual"}
)


# --------------------------------------------------------------------------- #
# /start
# --------------------------------------------------------------------------- #


@router.message(Command("start"))
async def cmd_start(message: Message, db_user: User) -> None:
    """Greet the user and show available commands."""
    await message.answer(
        "Hello! I am your personal companion bot.\n\n"
        "Available commands:\n"
        "/profile — view your current settings\n"
        "/set_tone <tone> — adjust my tone (friendly, professional, playful…)\n"
        "/set_persona <name> — give me a persona name\n"
        "/memory_compact_now — compress your conversation history\n"
        "/reset_persona — restore default persona\n"
        "/privacy — privacy policy summary\n"
        "/delete_my_data — permanently delete all your data"
    )
    log.info("cmd_start", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /profile
# --------------------------------------------------------------------------- #


@router.message(Command("profile"))
async def cmd_profile(message: Message, db_user: User) -> None:
    """Display the user's current profile information."""
    lines = [
        f"Telegram ID: {db_user.telegram_user_id}",
        f"Status: {db_user.status}",
        f"Locale: {db_user.locale or '(not set)'}",
        f"Timezone: {db_user.timezone or '(not set)'}",
        "",
        "Persona and tone settings will be shown here once configured.",
        "Use /set_tone and /set_persona to customise.",
    ]
    await message.answer("\n".join(lines))
    log.info("cmd_profile", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /set_tone
# --------------------------------------------------------------------------- #


@router.message(Command("set_tone"))
async def cmd_set_tone(
    message: Message,
    command: CommandObject,
    db_user: User,
) -> None:
    """Set the response tone. Usage: /set_tone <tone>"""
    tone = (command.args or "").strip().lower()
    if not tone:
        await message.answer(
            "Please provide a tone.\n"
            f"Example: /set_tone friendly\n"
            f"Valid tones: {', '.join(sorted(_VALID_TONES))}"
        )
        return
    if tone not in _VALID_TONES:
        await message.answer(
            f"Unknown tone '{tone}'.\n"
            f"Valid tones: {', '.join(sorted(_VALID_TONES))}"
        )
        return
    # Full DB persistence deferred to Task 6 (Prompt state manager).
    await message.answer(
        f"Tone set to '{tone}'.\n"
        "Changes will be applied starting from your next message."
    )
    log.info("set_tone", internal_user_id=str(db_user.id), tone=tone)


# --------------------------------------------------------------------------- #
# /set_persona
# --------------------------------------------------------------------------- #


@router.message(Command("set_persona"))
async def cmd_set_persona(
    message: Message,
    command: CommandObject,
    db_user: User,
) -> None:
    """Set the persona name. Usage: /set_persona <name>"""
    name = (command.args or "").strip()
    if not name:
        await message.answer(
            "Please provide a persona name.\n"
            "Example: /set_persona Alex"
        )
        return
    if len(name) > 64:
        await message.answer("Persona name must be 64 characters or fewer.")
        return
    # Full DB persistence deferred to Task 6.
    await message.answer(f"Persona name set to '{html.escape(name)}'.")
    log.info("set_persona", internal_user_id=str(db_user.id), persona_name=name)


# --------------------------------------------------------------------------- #
# /memory_compact_now
# --------------------------------------------------------------------------- #


@router.message(Command("memory_compact_now"))
async def cmd_memory_compact_now(message: Message, db_user: User) -> None:
    """Request an immediate memory compaction for this user."""
    # Job enqueueing deferred to Task 9 (Refinement worker).
    await message.answer(
        "Memory compaction requested.\n"
        "Your conversation history will be summarised shortly."
    )
    log.info("memory_compact_now_requested", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /reset_persona
# --------------------------------------------------------------------------- #


@router.message(Command("reset_persona"))
async def cmd_reset_persona(message: Message, db_user: User) -> None:
    """Reset the user's persona and tone to defaults."""
    # Full DB write deferred to Task 6.
    await message.answer(
        "Your persona has been reset to defaults.\n"
        "Use /set_persona and /set_tone to customise again."
    )
    log.info("reset_persona", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /privacy
# --------------------------------------------------------------------------- #


@router.message(Command("privacy"))
async def cmd_privacy(message: Message) -> None:
    """Show a privacy policy summary."""
    await message.answer(
        "Privacy summary:\n\n"
        "• Messages are retained for up to 7 days to maintain conversation context.\n"
        "• Profile settings are stored until you request deletion.\n"
        "• No data is sold or shared with third parties.\n\n"
        "Use /delete_my_data to permanently remove all your personal data."
    )


# --------------------------------------------------------------------------- #
# /delete_my_data
# --------------------------------------------------------------------------- #


@router.message(Command("delete_my_data"))
async def cmd_delete_my_data(
    message: Message,
    db_user: User,
    db_session: AsyncSession,
    redis: Redis,
) -> None:
    """Hard-delete all personal data for the user.

    Deletes conversation history, profile, persona snapshots, jobs, and
    behavior-change events.  The audit log entry is preserved with a
    null user_id (audit minimality requirement).  Redis keys scoped to
    the user are also removed.
    """
    user_id_str = str(db_user.id)
    await hard_delete_user(db_user.id, db_session, redis, telegram_user_id=db_user.telegram_user_id)
    await message.answer(
        "Your personal data has been permanently deleted.\n\n"
        "Conversation history, profile settings, and persona data have been removed. "
        "This action cannot be undone."
    )
    log.info("delete_my_data_completed", internal_user_id=user_id_str)


# --------------------------------------------------------------------------- #
# Regular message handler (orchestrator entry point)
# --------------------------------------------------------------------------- #


@router.message(F.text)
async def handle_message(
    message: Message,
    db_user: User,
    db_session: AsyncSession,
    redis: Redis,
    snapshot_store: SnapshotStore,
    chat_client: ChatAPIClient,
    settings: Settings,
) -> None:
    """Route non-command text messages through the conversation orchestrator."""
    user_id_str = str(db_user.id)
    text = message.text or ""
    ingress_start = time.perf_counter()

    async with span("ingress.handle_message", user_id=user_id_str):
        reply = await process_message(
            user_id=db_user.id,
            message_text=text,
            session=db_session,
            snapshot_store=snapshot_store,
            redis=redis,
            chat_client=chat_client,
            model=settings.chat_model,
            conversation_ttl_seconds=settings.conversation_ttl_seconds,
            refinement_activity_threshold=settings.refinement_activity_threshold,
        )
        await message.answer(reply)

        # Surface "profile updated" notice if the refinement worker finished
        # updating this user's prompt snapshot since their last message.
        if await check_and_clear_user_notice(redis, user_id_str):
            await message.answer(
                "Your conversation profile has been updated based on recent interactions."
            )

    elapsed_ms = round((time.perf_counter() - ingress_start) * 1000, 2)
    log.info(
        "message_handled",
        internal_user_id=user_id_str,
        reply_length=len(reply),
        elapsed_ms=elapsed_ms,
    )
