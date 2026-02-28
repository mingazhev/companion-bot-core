"""Command handlers for all registered bot commands.

Each handler is a plain async function registered on the shared *router*.
Handlers receive injected dependencies (``db_user``, ``tg_user``) populated
by :class:`~companion_bot_core.bot.middleware.IngressMiddleware`.

Commands implemented here
-------------------------
- /start              \u2014 welcome message
- /profile            \u2014 show current user settings
- /set_language       \u2014 switch chat language (ru|en)
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

from companion_bot_core.behavior.extractor import VALID_TONES
from companion_bot_core.db.models import User, UserProfile
from companion_bot_core.i18n import SUPPORTED_LOCALES, normalize_locale, tr
from companion_bot_core.logging_config import get_logger
from companion_bot_core.orchestrator import process_message
from companion_bot_core.privacy.field_encryption import NOOP_ENCRYPTOR, FieldEncryptor
from companion_bot_core.prompt.helpers import (
    acquire_profile_advisory_lock,
    get_or_create_profile,
    rebuild_and_save_snapshot,
)
from companion_bot_core.tracing import span

if TYPE_CHECKING:
    from aiogram.types import Message
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from companion_bot_core.config import Settings
    from companion_bot_core.inference.client import ChatAPIClient
    from companion_bot_core.prompt.snapshot_store import SnapshotStore

from companion_bot_core.privacy.delete_user import hard_delete_user
from companion_bot_core.prompt.postgres_store import PROFILE_LOCK_UNLOCK_SCRIPT, defer_lock_release
from companion_bot_core.prompt.rollback import RollbackError, rollback_to_previous
from companion_bot_core.redis.queues import enqueue_refinement_job
from companion_bot_core.refinement.worker import check_and_clear_user_notice

log = get_logger(__name__)

router = Router(name="commands")

# Telegram limits plain-text messages to 4096 characters.
_TG_MSG_LIMIT = 4096


def _user_locale(db_user: User | None) -> str:
    return normalize_locale(db_user.locale if db_user is not None else None)



# --------------------------------------------------------------------------- #
# /start
# --------------------------------------------------------------------------- #


@router.message(Command("start"))
async def cmd_start(message: Message, db_user: User) -> None:
    """Greet the user and show available commands."""
    await message.answer(tr("start.text", _user_locale(db_user)), parse_mode=None)
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
    locale = _user_locale(db_user)
    enc = encryptor or NOOP_ENCRYPTOR
    result = await db_session.execute(
        select(UserProfile).where(UserProfile.user_id == db_user.id)
    )
    profile = result.scalar_one_or_none()

    persona_display = tr("profile.not_set", locale)
    tone_display = tr("profile.not_set", locale)
    if profile is not None:
        if profile.persona_name:
            persona_display = enc.decrypt_safe(
                profile.persona_name,
                default=tr("profile.decrypt_failed", locale),
            )
        if profile.tone:
            tone_display = enc.decrypt_safe(
                profile.tone,
                default=tr("profile.decrypt_failed", locale),
            )

    await message.answer(
        tr(
            "profile.summary",
            locale,
            telegram_id=db_user.telegram_user_id,
            status=db_user.status,
            user_locale=normalize_locale(db_user.locale),
            timezone=db_user.timezone or tr("profile.not_set", locale),
            persona=persona_display,
            tone=tone_display,
        ),
        parse_mode=None,
    )
    log.info("cmd_profile", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /set_language
# --------------------------------------------------------------------------- #


@router.message(Command("set_language"))
async def cmd_set_language(
    message: Message,
    command: CommandObject,
    db_user: User,
    db_session: AsyncSession,
) -> None:
    """Set chat language. Usage: /set_language <ru|en>."""
    current_locale = _user_locale(db_user)
    raw = (command.args or "").strip().lower()
    if not raw:
        await message.answer(tr("set_language.help", current_locale), parse_mode=None)
        return

    if raw.startswith("ru"):
        new_locale = "ru"
    elif raw.startswith("en"):
        new_locale = "en"
    else:
        await message.answer(
            tr("set_language.invalid", current_locale, value=html.escape(raw)),
            parse_mode=None,
        )
        return

    if new_locale not in SUPPORTED_LOCALES:
        await message.answer(
            tr("set_language.invalid", current_locale, value=html.escape(raw)),
            parse_mode=None,
        )
        return

    db_user.locale = new_locale
    await db_session.flush()
    await message.answer(tr("set_language.updated", new_locale), parse_mode=None)
    log.info("set_language", internal_user_id=str(db_user.id), locale=new_locale)


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
    locale = _user_locale(db_user)
    enc = encryptor or NOOP_ENCRYPTOR
    tone = (command.args or "").strip().lower()
    if not tone:
        tones = ", ".join(sorted(VALID_TONES))
        await message.answer(
            tr("set_tone.missing", locale, tones=tones),
            parse_mode=None,
        )
        return
    if tone not in VALID_TONES:
        tones = ", ".join(sorted(VALID_TONES))
        await message.answer(
            tr("set_tone.invalid", locale, tone=html.escape(tone), tones=tones),
            parse_mode=None,
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
            await message.answer(tr("profile.lock_in_progress", locale), parse_mode=None)
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
            tr("set_tone.updated", locale, tone=tone),
            parse_mode=None,
        )
        log.info("set_tone", internal_user_id=str(db_user.id), tone=tone)
        # Defer release until after commit + Redis pointer flush (success path).
        if lock_held:
            # redis cannot be None here: lock_held is only True after a
            # successful redis.set() inside `if redis is not None`.
            defer_lock_release(db_session, lock_key, lock_token)
            deferred = True
    finally:
        if lock_held and not deferred:
            if redis is None:  # invariant: should never happen
                log.error("profile_lock_release_impossible", internal_user_id=str(db_user.id))
            else:
                try:
                    await redis.eval(PROFILE_LOCK_UNLOCK_SCRIPT, 1, lock_key, lock_token)  # type: ignore[misc]
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
    locale = _user_locale(db_user)
    enc = encryptor or NOOP_ENCRYPTOR
    name = (command.args or "").strip()
    if not name:
        await message.answer(tr("set_persona.missing", locale), parse_mode=None)
        return
    if len(name) > 64:
        await message.answer(tr("set_persona.too_long", locale), parse_mode=None)
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
        await message.answer(tr("set_persona.control_chars", locale), parse_mode=None)
        return

    # Serialize concurrent profile updates for the same user.
    lock_key = f"profile:write:{db_user.id}"
    lock_token = secrets.token_hex(16)
    lock_held = False
    if redis is not None:
        lock_held = bool(await redis.set(lock_key, lock_token, nx=True, ex=120))
        if not lock_held:
            await message.answer(tr("profile.lock_in_progress", locale), parse_mode=None)
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
        await message.answer(
            tr("set_persona.updated", locale, name=html.escape(name)),
            parse_mode=None,
        )
        log.info("set_persona", internal_user_id=str(db_user.id), persona_name=name)
        # Defer release until after commit + Redis pointer flush (success path).
        if lock_held:
            # redis cannot be None here: lock_held is only True after a
            # successful redis.set() inside `if redis is not None`.
            defer_lock_release(db_session, lock_key, lock_token)
            deferred = True
    finally:
        if lock_held and not deferred:
            if redis is None:  # invariant: should never happen
                log.error("profile_lock_release_impossible", internal_user_id=str(db_user.id))
            else:
                try:
                    await redis.eval(PROFILE_LOCK_UNLOCK_SCRIPT, 1, lock_key, lock_token)  # type: ignore[misc]
                except Exception:  # noqa: BLE001
                    log.warning("profile_lock_release_failed", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /memory_compact_now
# --------------------------------------------------------------------------- #


@router.message(Command("memory_compact_now"))
async def cmd_memory_compact_now(message: Message, db_user: User, redis: Redis) -> None:
    """Request an immediate memory compaction for this user."""
    locale = _user_locale(db_user)
    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)

    # Acquire the shared dedup guard to prevent flooding the refinement queue.
    guard_key = f"refinement:pending:{user_id_str}"
    acquired = await redis.set(guard_key, "1", nx=True, ex=600)
    if not acquired:
        await message.answer(tr("memory_compact.in_progress", locale), parse_mode=None)
        return

    try:
        await enqueue_refinement_job(redis, user_id_str, {"trigger": "manual_compact"})
    except Exception:  # noqa: BLE001
        try:
            await redis.delete(guard_key)
        except Exception:  # noqa: BLE001
            log.warning("compact_guard_cleanup_failed", internal_user_id=user_id_str)
        await message.answer(tr("memory_compact.enqueue_failed", locale), parse_mode=None)
        log.warning("memory_compact_enqueue_failed", internal_user_id=user_id_str)
        return

    await message.answer(tr("memory_compact.requested", locale), parse_mode=None)
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
    locale = _user_locale(db_user)
    # Serialize concurrent profile updates for the same user.
    lock_key = f"profile:write:{db_user.id}"
    lock_token = secrets.token_hex(16)
    lock_held = False
    if redis is not None:
        lock_held = bool(await redis.set(lock_key, lock_token, nx=True, ex=120))
        if not lock_held:
            await message.answer(tr("profile.lock_in_progress", locale), parse_mode=None)
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
        await message.answer(tr("reset_persona.updated", locale), parse_mode=None)
        log.info("reset_persona", internal_user_id=str(db_user.id))
        # Defer release until after commit + Redis pointer flush (success path).
        if lock_held:
            # redis cannot be None here: lock_held is only True after a
            # successful redis.set() inside `if redis is not None`.
            defer_lock_release(db_session, lock_key, lock_token)
            deferred = True
    finally:
        if lock_held and not deferred:
            if redis is None:  # invariant: should never happen
                log.error("profile_lock_release_impossible", internal_user_id=str(db_user.id))
            else:
                try:
                    await redis.eval(PROFILE_LOCK_UNLOCK_SCRIPT, 1, lock_key, lock_token)  # type: ignore[misc]
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
    locale = _user_locale(db_user)
    try:
        rolled_back = await rollback_to_previous(
            snapshot_store, db_user.id, session=db_session,
        )
    except RollbackError as exc:
        await message.answer(str(exc), parse_mode=None)
        return
    await message.answer(
        tr("rollback.updated", locale, version=rolled_back.version),
        parse_mode=None,
    )
    log.info("cmd_rollback", internal_user_id=str(db_user.id), version=rolled_back.version)


# --------------------------------------------------------------------------- #
# /privacy
# --------------------------------------------------------------------------- #


@router.message(Command("privacy"))
async def cmd_privacy(message: Message, db_user: User | None = None) -> None:
    """Show a privacy policy summary."""
    await message.answer(tr("privacy.summary", _user_locale(db_user)), parse_mode=None)


# --------------------------------------------------------------------------- #
# /delete_my_data
# --------------------------------------------------------------------------- #


@router.message(Command("delete_my_data"))
async def cmd_delete_my_data(
    message: Message,
    db_user: User,
    db_session: AsyncSession,
    redis: Redis,
    snapshot_store: SnapshotStore,
) -> None:
    """Hard-delete all personal data for the user.

    Deletes conversation history, profile, persona snapshots, jobs, and
    behavior-change events.  The audit log entry is preserved with a
    null user_id (audit minimality requirement).  Redis keys scoped to
    the user are also removed.  In-memory snapshot data is also purged.
    """
    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)
    await hard_delete_user(db_user.id, db_session, redis, telegram_user_id=db_user.telegram_user_id)
    await snapshot_store.delete_for_user(db_user.id)
    log.info("delete_my_data_completed", internal_user_id=user_id_str)
    try:
        await message.answer(tr("delete_my_data.done", locale), parse_mode=None)
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
    redis: Redis,
    snapshot_store: SnapshotStore,
    chat_client: ChatAPIClient,
    settings: Settings,
    encryptor: FieldEncryptor | None = None,
) -> None:
    """Route non-command text messages through the conversation orchestrator."""
    locale = _user_locale(db_user)
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
                locale=locale,
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
                    tr("handle.error", locale),
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
                await message.answer(tr("notice.profile_updated", locale), parse_mode=None)
        except Exception:
            log.warning("notice_send_failed", internal_user_id=user_id_str)

    elapsed_ms = round((time.perf_counter() - ingress_start) * 1000, 2)
    log.info(
        "message_handled",
        internal_user_id=user_id_str,
        reply_length=len(reply),
        elapsed_ms=elapsed_ms,
    )
