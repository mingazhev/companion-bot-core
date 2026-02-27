"""Command handlers for all registered bot commands.

Each handler is a plain async function registered on the shared *router*.
Handlers receive injected dependencies (``db_user``, ``tg_user``) populated
by :class:`~tdbot.bot.middleware.IngressMiddleware`.

Commands implemented here
-------------------------
- /start              \u2014 welcome message
- /profile            \u2014 show current user settings
- /set_tone           \u2014 update response tone (persists to DB + rebuilds snapshot)
- /set_persona        \u2014 update persona name (persists to DB + rebuilds snapshot)
- /memory_compact_now \u2014 trigger memory compaction job
- /reset_persona      \u2014 restore default persona (persists to DB + rebuilds snapshot)
- /rollback           \u2014 revert active prompt snapshot to the previous version
- /privacy            \u2014 display privacy policy summary
- /delete_my_data     \u2014 initiate hard-delete flow
"""

from __future__ import annotations

import html
import secrets
import time
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from sqlalchemy import select

from tdbot.behavior.extractor import VALID_TONES
from tdbot.db.models import UserProfile
from tdbot.logging_config import get_logger
from tdbot.orchestrator import process_message
from tdbot.privacy.field_encryption import NOOP_ENCRYPTOR, FieldEncryptor
from tdbot.prompt.helpers import (
    acquire_profile_advisory_lock,
    get_or_create_profile,
    rebuild_and_save_snapshot,
)
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
from tdbot.prompt.postgres_store import PROFILE_LOCK_UNLOCK_SCRIPT, defer_lock_release
from tdbot.prompt.rollback import RollbackError, rollback_to_previous
from tdbot.redis.queues import enqueue_refinement_job
from tdbot.refinement.worker import check_and_clear_user_notice

log = get_logger(__name__)

router = Router(name="commands")

# Telegram limits plain-text messages to 4096 characters.
_TG_MSG_LIMIT = 4096



# --------------------------------------------------------------------------- #
# /start
# --------------------------------------------------------------------------- #


@router.message(Command("start"))
async def cmd_start(message: Message, db_user: User) -> None:
    """Greet the user and show available commands."""
    await message.answer(
        "Hello! I am your personal companion bot.\n\n"
        "Available commands:\n"
        "/profile \u2014 view your current settings\n"
        "/set_tone <tone> \u2014 adjust my tone (friendly, professional, playful\u2026)\n"
        "/set_persona <name> \u2014 give me a persona name\n"
        "/memory_compact_now \u2014 compress your conversation history\n"
        "/reset_persona \u2014 restore default persona\n"
        "/rollback \u2014 revert to the previous prompt version\n"
        "/privacy \u2014 privacy policy summary\n"
        "/delete_my_data \u2014 permanently delete all your data"
    )
    log.info("cmd_start", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /profile
# --------------------------------------------------------------------------- #


@router.message(Command("profile"))
async def cmd_profile(
    message: Message,
    db_user: User,
    db_session: AsyncSession,
    encryptor: FieldEncryptor | None = None,
) -> None:
    """Display the user's current profile information."""
    enc = encryptor or NOOP_ENCRYPTOR
    result = await db_session.execute(
        select(UserProfile).where(UserProfile.user_id == db_user.id)
    )
    profile = result.scalar_one_or_none()

    persona_display = "(not set)"
    tone_display = "(not set)"
    if profile is not None:
        if profile.persona_name:
            persona_display = enc.decrypt_safe(profile.persona_name, default="(unable to decrypt)")
        if profile.tone:
            tone_display = enc.decrypt_safe(profile.tone, default="(unable to decrypt)")

    lines = [
        f"Telegram ID: {db_user.telegram_user_id}",
        f"Status: {db_user.status}",
        f"Locale: {db_user.locale or '(not set)'}",
        f"Timezone: {db_user.timezone or '(not set)'}",
        "",
        f"Persona: {persona_display}",
        f"Tone: {tone_display}",
        "",
        "Use /set_tone and /set_persona to customise.",
    ]
    await message.answer("\n".join(lines), parse_mode=None)
    log.info("cmd_profile", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /set_tone
# --------------------------------------------------------------------------- #


@router.message(Command("set_tone"))
async def cmd_set_tone(
    message: Message,
    command: CommandObject,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    encryptor: FieldEncryptor | None = None,
    redis: Redis | None = None,
) -> None:
    """Set the response tone. Usage: /set_tone <tone>"""
    enc = encryptor or NOOP_ENCRYPTOR
    tone = (command.args or "").strip().lower()
    if not tone:
        await message.answer(
            "Please provide a tone.\n"
            f"Example: /set_tone friendly\n"
            f"Valid tones: {', '.join(sorted(VALID_TONES))}"
        )
        return
    if tone not in VALID_TONES:
        await message.answer(
            f"Unknown tone '{html.escape(tone)}'.\n"
            f"Valid tones: {', '.join(sorted(VALID_TONES))}"
        )
        return

    # Serialize concurrent profile updates for the same user to prevent a
    # read-modify-write race where two simultaneous commands each read the
    # same base snapshot and the last commit silently drops the other change.
    lock_key = f"profile:write:{db_user.id}"
    lock_token = secrets.token_hex(16)
    lock_held = False
    if redis is not None:
        lock_held = bool(await redis.set(lock_key, lock_token, nx=True, ex=120))
        if not lock_held:
            await message.answer(
                "A profile update is already in progress. Please try again in a moment."
            )
            return

    # When the success path completes, the lock release is deferred until after
    # the DB transaction commits and deferred Redis pointer writes are flushed
    # (handled by IngressMiddleware).  On the error path (exception before
    # deferral), the lock is released immediately in the finally block so the
    # user can retry without waiting for the TTL to expire.
    deferred = False
    try:
        # Acquire a PostgreSQL advisory transaction lock to guarantee
        # serialization for the full duration of the DB transaction.  This
        # complements the Redis TTL lock above: the Redis lock provides a fast
        # rejection for obvious concurrency; the advisory lock eliminates the
        # residual race where the TTL could expire before the transaction
        # commits.  The advisory lock auto-releases when the transaction ends.
        await acquire_profile_advisory_lock(db_session, db_user.id)
        profile = await get_or_create_profile(db_session, db_user.id)
        # Decrypt existing persona_name for prompt building before encrypting new tone.
        raw_persona = (
            enc.decrypt_safe(profile.persona_name, default="")
            if profile.persona_name else None
        )
        profile.tone = enc.encrypt(tone)
        await db_session.flush()
        await rebuild_and_save_snapshot(
            snapshot_store, db_user.id, raw_persona, tone,
            session=db_session,
        )
        await message.answer(
            f"Tone set to '{tone}'.\n"
            "Changes will be applied starting from your next message."
        )
        log.info("set_tone", internal_user_id=str(db_user.id), tone=tone)
        # Defer release until after commit + Redis pointer flush (success path).
        if lock_held:
            assert redis is not None  # lock_held is True only when redis was used
            defer_lock_release(db_session, lock_key, lock_token)
            deferred = True
    finally:
        if lock_held and not deferred:
            assert redis is not None  # lock_held is True only when redis was used
            try:
                await redis.eval(PROFILE_LOCK_UNLOCK_SCRIPT, 1, lock_key, lock_token)  # type: ignore[no-untyped-call]
            except Exception:  # noqa: BLE001
                log.warning("profile_lock_release_failed", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /set_persona
# --------------------------------------------------------------------------- #


@router.message(Command("set_persona"))
async def cmd_set_persona(
    message: Message,
    command: CommandObject,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    encryptor: FieldEncryptor | None = None,
    redis: Redis | None = None,
) -> None:
    """Set the persona name. Usage: /set_persona <name>"""
    enc = encryptor or NOOP_ENCRYPTOR
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
    # Reject control characters (newlines, tabs, DEL, Unicode direction overrides,
    # zero-width chars) to prevent stored prompt injection via persona names.
    _banned_chars = frozenset(
        "\x7f"                          # DEL
        "\u200b\u200c\u200d\u200e\u200f"  # zero-width + directional marks
        "\u2028\u2029"                  # line/paragraph separators
        "\u202a\u202b\u202c\u202d\u202e"  # bidi embedding/override
        "\ufeff"                        # BOM / zero-width no-break space
    )
    if any(c < " " or c in _banned_chars for c in name):
        await message.answer("Persona name must not contain control characters.")
        return

    # Serialize concurrent profile updates for the same user.
    lock_key = f"profile:write:{db_user.id}"
    lock_token = secrets.token_hex(16)
    lock_held = False
    if redis is not None:
        lock_held = bool(await redis.set(lock_key, lock_token, nx=True, ex=120))
        if not lock_held:
            await message.answer(
                "A profile update is already in progress. Please try again in a moment."
            )
            return

    deferred = False
    try:
        await acquire_profile_advisory_lock(db_session, db_user.id)
        profile = await get_or_create_profile(db_session, db_user.id)
        # Decrypt existing tone for prompt building before encrypting new persona name.
        raw_tone = (
            enc.decrypt_safe(profile.tone, default="")
            if profile.tone else None
        )
        profile.persona_name = enc.encrypt(name)
        await db_session.flush()
        await rebuild_and_save_snapshot(
            snapshot_store, db_user.id, name, raw_tone,
            session=db_session,
        )
        await message.answer(f"Persona name set to '{html.escape(name)}'.")
        log.info("set_persona", internal_user_id=str(db_user.id), persona_name=name)
        # Defer release until after commit + Redis pointer flush (success path).
        if lock_held:
            assert redis is not None  # lock_held is True only when redis was used
            defer_lock_release(db_session, lock_key, lock_token)
            deferred = True
    finally:
        if lock_held and not deferred:
            assert redis is not None  # lock_held is True only when redis was used
            try:
                await redis.eval(PROFILE_LOCK_UNLOCK_SCRIPT, 1, lock_key, lock_token)  # type: ignore[no-untyped-call]
            except Exception:  # noqa: BLE001
                log.warning("profile_lock_release_failed", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /memory_compact_now
# --------------------------------------------------------------------------- #


@router.message(Command("memory_compact_now"))
async def cmd_memory_compact_now(message: Message, db_user: User, redis: Redis) -> None:  # type: ignore[type-arg]
    """Request an immediate memory compaction for this user."""
    user_id_str = str(db_user.id)

    # Acquire the shared dedup guard to prevent flooding the refinement queue.
    guard_key = f"refinement:pending:{user_id_str}"
    acquired = await redis.set(guard_key, "1", nx=True, ex=600)
    if not acquired:
        await message.answer("A compaction is already in progress. Please wait.")
        return

    try:
        await enqueue_refinement_job(redis, user_id_str, {"trigger": "manual_compact"})
    except Exception:  # noqa: BLE001
        try:
            await redis.delete(guard_key)
        except Exception:  # noqa: BLE001
            log.warning("compact_guard_cleanup_failed", internal_user_id=user_id_str)
        await message.answer("Failed to enqueue compaction. Please try again.")
        log.warning("memory_compact_enqueue_failed", internal_user_id=user_id_str)
        return

    await message.answer(
        "Memory compaction requested.\n"
        "Your prompt profile will be refined based on recent conversations shortly."
    )
    log.info("memory_compact_now_requested", internal_user_id=user_id_str)


# --------------------------------------------------------------------------- #
# /reset_persona
# --------------------------------------------------------------------------- #


@router.message(Command("reset_persona"))
async def cmd_reset_persona(
    message: Message,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    redis: Redis | None = None,
) -> None:
    """Reset the user's persona and tone to defaults."""
    # Serialize concurrent profile updates for the same user.
    lock_key = f"profile:write:{db_user.id}"
    lock_token = secrets.token_hex(16)
    lock_held = False
    if redis is not None:
        lock_held = bool(await redis.set(lock_key, lock_token, nx=True, ex=120))
        if not lock_held:
            await message.answer(
                "A profile update is already in progress. Please try again in a moment."
            )
            return

    deferred = False
    try:
        await acquire_profile_advisory_lock(db_session, db_user.id)
        profile = await get_or_create_profile(db_session, db_user.id)
        profile.persona_name = None
        profile.tone = None
        await db_session.flush()
        await rebuild_and_save_snapshot(
            snapshot_store, db_user.id, None, None,
            session=db_session,
        )
        await message.answer(
            "Your persona has been reset to defaults.\n"
            "Use /set_persona and /set_tone to customise again."
        )
        log.info("reset_persona", internal_user_id=str(db_user.id))
        # Defer release until after commit + Redis pointer flush (success path).
        if lock_held:
            assert redis is not None  # lock_held is True only when redis was used
            defer_lock_release(db_session, lock_key, lock_token)
            deferred = True
    finally:
        if lock_held and not deferred:
            assert redis is not None  # lock_held is True only when redis was used
            try:
                await redis.eval(PROFILE_LOCK_UNLOCK_SCRIPT, 1, lock_key, lock_token)  # type: ignore[no-untyped-call]
            except Exception:  # noqa: BLE001
                log.warning("profile_lock_release_failed", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /rollback
# --------------------------------------------------------------------------- #


@router.message(Command("rollback"))
async def cmd_rollback(
    message: Message,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
) -> None:
    """Revert the active prompt snapshot to the previous version."""
    try:
        rolled_back = await rollback_to_previous(
            snapshot_store, db_user.id, session=db_session,
        )
    except RollbackError as exc:
        await message.answer(str(exc))
        return
    await message.answer(
        f"Prompt rolled back to version {rolled_back.version}.\n"
        "Changes will be applied starting from your next message."
    )
    log.info("cmd_rollback", internal_user_id=str(db_user.id), version=rolled_back.version)


# --------------------------------------------------------------------------- #
# /privacy
# --------------------------------------------------------------------------- #


@router.message(Command("privacy"))
async def cmd_privacy(message: Message) -> None:
    """Show a privacy policy summary."""
    await message.answer(
        "Privacy summary:\n\n"
        "\u2022 Messages are retained for up to 7 days to maintain conversation context.\n"
        "\u2022 Profile settings are stored until you request deletion.\n"
        "\u2022 No data is sold or shared with third parties.\n\n"
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
    redis: Redis,  # type: ignore[type-arg]
    snapshot_store: SnapshotStore,
) -> None:
    """Hard-delete all personal data for the user.

    Deletes conversation history, profile, persona snapshots, jobs, and
    behavior-change events.  The audit log entry is preserved with a
    null user_id (audit minimality requirement).  Redis keys scoped to
    the user are also removed.  In-memory snapshot data is also purged.
    """
    user_id_str = str(db_user.id)
    await hard_delete_user(db_user.id, db_session, redis, telegram_user_id=db_user.telegram_user_id)
    await snapshot_store.delete_for_user(db_user.id)
    log.info("delete_my_data_completed", internal_user_id=user_id_str)
    try:
        await message.answer(
            "Your personal data has been permanently deleted.\n\n"
            "Conversation history, profile settings, and persona data have been removed. "
            "This action cannot be undone."
        )
    except Exception:  # noqa: BLE001
        log.warning("delete_my_data_confirmation_send_failed", internal_user_id=user_id_str)


# --------------------------------------------------------------------------- #
# Regular message handler (orchestrator entry point)
# --------------------------------------------------------------------------- #


@router.message(F.text)
async def handle_message(
    message: Message,
    db_user: User,
    db_session: AsyncSession,
    redis: Redis,  # type: ignore[type-arg]
    snapshot_store: SnapshotStore,
    chat_client: ChatAPIClient,
    settings: Settings,
    encryptor: FieldEncryptor | None = None,
) -> None:
    """Route non-command text messages through the conversation orchestrator."""
    user_id_str = str(db_user.id)
    text = message.text or ""
    ingress_start = time.perf_counter()
    reply = ""

    async with span("ingress.handle_message", user_id=user_id_str):
        try:
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
                refinement_cadence_seconds=settings.refinement_cadence_seconds,
                encryptor=encryptor,
            )
        except Exception:
            log.exception(
                "process_message_failed",
                internal_user_id=user_id_str,
            )
            # Send a user-facing error reply before re-raising so the
            # middleware rolls back the DB transaction (prevents committing
            # partially-flushed state such as assistant messages the user
            # never saw).
            try:
                await message.answer(
                    "Something went wrong while processing your message. "
                    "Please try again in a moment.",
                    parse_mode=None,
                )
            except Exception:  # noqa: BLE001
                log.warning("error_reply_send_failed", internal_user_id=user_id_str)
            raise

        # Split if the reply exceeds Telegram's message-length limit.
        # Wrap in try/except so a Telegram API failure (timeout, rate limit)
        # does not propagate and roll back the DB transaction \u2014 Redis state
        # (activity counter, refinement jobs) is already committed.
        try:
            for i in range(0, len(reply), _TG_MSG_LIMIT):
                await message.answer(reply[i : i + _TG_MSG_LIMIT], parse_mode=None)
        except Exception as exc:
            log.warning(
                "reply_send_failed",
                internal_user_id=user_id_str,
                reply_length=len(reply),
                error=str(exc),
            )

        # Surface "profile updated" notice if the refinement worker finished
        # updating this user's prompt snapshot since their last message.
        try:
            if await check_and_clear_user_notice(redis, user_id_str):
                await message.answer(
                    "Your conversation profile has been updated based on recent interactions."
                )
        except Exception:
            log.warning("notice_send_failed", internal_user_id=user_id_str)

    elapsed_ms = round((time.perf_counter() - ingress_start) * 1000, 2)
    log.info(
        "message_handled",
        internal_user_id=user_id_str,
        reply_length=len(reply),
        elapsed_ms=elapsed_ms,
    )
