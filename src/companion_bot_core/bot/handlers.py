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
- /reset              \u2014 wipe child data and restart onboarding
"""

from __future__ import annotations

import asyncio
import html
import json
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aiogram import F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import delete, select

from companion_bot_core.behavior.extractor import VALID_TONES
from companion_bot_core.db.models import (
    BehaviorChangeEvent,
    ConversationMessage,
    Job,
    MemoryCompaction,
    PromptSnapshot,
    User,
    UserProfile,
)
from companion_bot_core.i18n import SUPPORTED_LOCALES, normalize_locale, tr
from companion_bot_core.logging_config import get_logger
from companion_bot_core.orchestrator import process_message
from companion_bot_core.policy.guardrails import check_prompt_injection, check_unsafe_role_change
from companion_bot_core.privacy.field_encryption import NOOP_ENCRYPTOR, FieldEncryptor
from companion_bot_core.prompt.helpers import (
    acquire_profile_advisory_lock,
    add_fact_to_profile,
    extract_memory_sections,
    get_or_create_profile,
    rebuild_and_save_snapshot,
    remove_fact_from_profile,
)
from companion_bot_core.tracing import span

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from aiogram.types import Message
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from companion_bot_core.config import Settings
    from companion_bot_core.inference.client import ChatAPIClient
    from companion_bot_core.prompt.snapshot_store import SnapshotStore

from companion_bot_core.dev.seeds import PERSONAS as SEED_PERSONAS
from companion_bot_core.privacy.delete_user import hard_delete_user
from companion_bot_core.prompt.merge_builder import (
    build_system_prompt as _build_system_prompt,
)
from companion_bot_core.prompt.merge_builder import (
    extract_base_template as _extract_base_template,
)
from companion_bot_core.prompt.merge_builder import (
    extract_section as _extract_section,
)
from companion_bot_core.prompt.postgres_store import PROFILE_LOCK_UNLOCK_SCRIPT, defer_lock_release
from companion_bot_core.prompt.rollback import RollbackError, rollback_to_previous
from companion_bot_core.prompt.schemas import (
    DEFAULT_SYSTEM_TEMPLATE as _DEFAULT_SYSTEM_TEMPLATE,
)
from companion_bot_core.prompt.schemas import (
    PromptComponents as _PromptComponents,
)
from companion_bot_core.prompt.schemas import (
    SnapshotRecord as _SnapshotRecord,
)
from companion_bot_core.redis.queues import enqueue_refinement_job
from companion_bot_core.refinement.worker import check_and_clear_user_notice

log = get_logger(__name__)

router = Router(name="commands")

# Telegram limits plain-text messages to 4096 characters.
_TG_MSG_LIMIT = 4096


def _user_locale(db_user: User | None) -> str:
    return normalize_locale(db_user.locale if db_user is not None else None)


def _is_group_chat(message: Message) -> bool:
    """Return True if the message originates from a group or supergroup."""
    return message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)


async def _guard_private_only(message: Message, db_user: User) -> bool:
    """Reply with an error and return True if called from a group chat."""
    if _is_group_chat(message):
        locale = _user_locale(db_user)
        await message.reply(
            tr("group.personal_command", locale), parse_mode=None,
        )
        return True
    return False


# Characters banned in user-supplied names (same set as cmd_set_persona).
_BANNED_NAME_CHARS: frozenset[str] = frozenset(
    "\x7f\u200b\u200c\u200d\u200e\u200f"
    "\u2028\u2029\u202a\u202b\u202c\u202d\u202e\ufeff"
)


def _sanitize_onboarding_name(text: str) -> str | None:
    """Validate and sanitize a name entered during onboarding.

    Returns the cleaned name or ``None`` if invalid.
    """
    name = text.strip()
    if not name or len(name) > 64:  # noqa: PLR2004
        return None
    if any(c < " " or c in _BANNED_NAME_CHARS for c in name):
        return None
    return name


def _interest_keyboard(locale: str) -> InlineKeyboardMarkup:
    """Build the interest-selection inline keyboard for onboarding step 2."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text=tr("interest.tech", locale),
                callback_data="onboard_interest:tech",
            ),
            InlineKeyboardButton(
                text=tr("interest.creative", locale),
                callback_data="onboard_interest:creative",
            ),
        ],
        [
            InlineKeyboardButton(
                text=tr("interest.learning", locale),
                callback_data="onboard_interest:learning",
            ),
            InlineKeyboardButton(
                text=tr("interest.fitness", locale),
                callback_data="onboard_interest:fitness",
            ),
        ],
        [
            InlineKeyboardButton(
                text=tr("onboarding.skip", locale),
                callback_data="onboard_interest:skip",
            ),
        ],
    ])


class _ProfileLockBusyError(Exception):
    """Raised when the Redis profile write lock cannot be acquired."""


@asynccontextmanager
async def _profile_write_lock(
    redis: Redis | None,
    db_session: AsyncSession,
    user_id: Any,
) -> AsyncGenerator[None, None]:
    """Acquire a Redis write lock for profile updates.

    On the success path the lock release is deferred until after the DB
    transaction commits (via ``defer_lock_release``).  On error the lock is
    released immediately so the user can retry without waiting for the TTL.

    Raises ``_ProfileLockBusyError`` when the lock is already held.
    """
    lock_key = f"profile:write:{user_id}"
    lock_token = secrets.token_hex(16)
    lock_held = False
    if redis is not None:
        lock_held = bool(await redis.set(lock_key, lock_token, nx=True, ex=120))
        if not lock_held:
            raise _ProfileLockBusyError

    deferred = False
    try:
        yield
        if lock_held:
            defer_lock_release(db_session, lock_key, lock_token)
            deferred = True
    finally:
        if lock_held and not deferred:
            if redis is None:  # pragma: no cover – invariant
                log.error("profile_lock_release_impossible")
            else:
                try:
                    await redis.eval(  # type: ignore[no-untyped-call]
                        PROFILE_LOCK_UNLOCK_SCRIPT, 1, lock_key, lock_token,
                    )
                except Exception:  # noqa: BLE001
                    log.warning("profile_lock_release_failed")


# --------------------------------------------------------------------------- #
# /start
# --------------------------------------------------------------------------- #


@router.message(Command("start"))
async def cmd_start(
    message: Message,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    redis: Redis,
    encryptor: FieldEncryptor | None = None,
) -> None:
    """Welcome message: value-prop for new users, personalised greeting for returning."""
    if _is_group_chat(message):
        await message.reply(tr("start.group_hint", _user_locale(db_user)), parse_mode=None)
        return
    locale = _user_locale(db_user)
    enc = encryptor or NOOP_ENCRYPTOR

    # Check if user has a profile (returning user)
    result = await db_session.execute(
        select(UserProfile).where(UserProfile.user_id == db_user.id)
    )
    profile = result.scalar_one_or_none()

    if profile is not None and (profile.persona_name or profile.tone):
        # Returning user — personalised greeting
        name = ""
        if profile.persona_name:
            name = enc.decrypt_safe(profile.persona_name, default="")
        if name:
            text = tr("start.welcome_back", locale, name=html.escape(name))
        else:
            text = tr("start.welcome_back_no_name", locale)
        await message.answer(text, parse_mode=None)
    else:
        # New user — warm value-prop + onboarding step 1 (name input)
        await message.answer(
            tr("start.welcome_new", locale), parse_mode=None,
        )
        state_key = f"{_ONBOARDING_PREFIX}:{db_user.id}"
        await redis.set(
            state_key, json.dumps({"step": "name"}), ex=_ONBOARDING_TTL,
        )
        await message.answer(
            tr("onboarding.step1_name", locale), parse_mode=None,
        )

    log.info("cmd_start", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /help
# --------------------------------------------------------------------------- #


@router.message(Command("help"))
async def cmd_help(message: Message, db_user: User) -> None:
    """Show available commands."""
    await message.answer(tr("help.text", _user_locale(db_user)), parse_mode=None)
    log.info("cmd_help", internal_user_id=str(db_user.id))


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
    if await _guard_private_only(message, db_user):
        return
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
            persona=persona_display,
            tone=tone_display,
        ),
        parse_mode=None,
    )
    log.info("cmd_profile", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /memory
# --------------------------------------------------------------------------- #


@router.message(Command("memory"))
async def cmd_memory(
    message: Message,
    db_user: User,
    snapshot_store: SnapshotStore,
) -> None:
    """Show what the bot remembers about the user."""
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    snapshot = await snapshot_store.get_active(db_user.id)
    sections = extract_memory_sections(snapshot)

    parts: list[str] = []
    has_content = False

    if sections["persona"]:
        parts.append(tr("memory.persona", locale, value=sections["persona"]))
        has_content = True
    if sections["tone"]:
        parts.append(tr("memory.tone", locale, value=sections["tone"]))
        has_content = True
    if sections["skills"]:
        parts.append(tr("memory.skills", locale, value=sections["skills"]))
        has_content = True

    profile_text = sections["long_term_profile"]
    if profile_text.strip():
        parts.append("")  # blank line separator
        parts.append(tr("memory.profile", locale, value=profile_text))
        has_content = True

    if has_content:
        text = tr("memory.header", locale) + "\n".join(parts)
    else:
        text = tr("memory.empty", locale)

    await message.answer(text, parse_mode=None)
    log.info("cmd_memory", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /remember
# --------------------------------------------------------------------------- #


@router.message(Command("remember"))
async def cmd_remember(
    message: Message,
    command: CommandObject,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
) -> None:
    """Append a fact to the user's long-term profile."""
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    fact = (command.args or "").strip()
    if not fact:
        await message.answer(tr("remember.missing", locale), parse_mode=None)
        return
    if len(fact) > 500:
        fact = fact[:500]
        await message.answer(
            tr("remember.truncated", locale), parse_mode=None,
        )

    # Guardrail: reject prompt injection in user-supplied facts.
    injection = check_prompt_injection(fact)
    if not injection.allowed:
        await message.answer(tr("guardrail.command_blocked", locale), parse_mode=None)
        log.warning(
            "remember_guardrail_blocked",
            internal_user_id=str(db_user.id),
            violation=injection.violation,
        )
        return

    await acquire_profile_advisory_lock(db_session, db_user.id)
    await add_fact_to_profile(snapshot_store, db_user.id, fact, session=db_session)
    await message.answer(
        tr("remember.saved", locale, fact=html.escape(fact)),
        parse_mode=None,
    )
    log.info("cmd_remember", internal_user_id=str(db_user.id))


# --------------------------------------------------------------------------- #
# /forget
# --------------------------------------------------------------------------- #


@router.message(Command("forget"))
async def cmd_forget(
    message: Message,
    command: CommandObject,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
) -> None:
    """Remove a matching fact from the user's long-term profile."""
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    query = (command.args or "").strip()
    if not query:
        await message.answer(tr("forget.missing", locale), parse_mode=None)
        return

    await acquire_profile_advisory_lock(db_session, db_user.id)
    removed = await remove_fact_from_profile(
        snapshot_store, db_user.id, query, session=db_session,
    )
    if removed is None:
        await message.answer(tr("forget.not_found", locale), parse_mode=None)
    else:
        await message.answer(
            tr("forget.done", locale, fact=html.escape(removed)),
            parse_mode=None,
        )
    log.info("cmd_forget", internal_user_id=str(db_user.id), found=removed is not None)


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
    if await _guard_private_only(message, db_user):
        return
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

    try:
        async with _profile_write_lock(redis, db_session, db_user.id):
            await acquire_profile_advisory_lock(db_session, db_user.id)
            profile = await get_or_create_profile(db_session, db_user.id)
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
    except _ProfileLockBusyError:
        await message.answer(tr("profile.lock_in_progress", locale), parse_mode=None)


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
    if await _guard_private_only(message, db_user):
        return
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

    # Guardrail check: reject prompt injection and unsafe role names.
    for _check in (check_prompt_injection, check_unsafe_role_change):
        result = _check(name)
        if not result.allowed:
            await message.answer(tr("guardrail.command_blocked", locale), parse_mode=None)
            log.warning(
                "set_persona_guardrail_blocked",
                internal_user_id=str(db_user.id),
                violation=result.violation,
            )
            return

    try:
        async with _profile_write_lock(redis, db_session, db_user.id):
            await acquire_profile_advisory_lock(db_session, db_user.id)
            profile = await get_or_create_profile(db_session, db_user.id)
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
    except _ProfileLockBusyError:
        await message.answer(tr("profile.lock_in_progress", locale), parse_mode=None)


# --------------------------------------------------------------------------- #
# /refresh_memory (renamed from /memory_compact_now)
# --------------------------------------------------------------------------- #


@router.message(Command("refresh_memory"))
async def cmd_refresh_memory(message: Message, db_user: User, redis: Redis) -> None:
    """Review conversations and refresh what the bot remembers about the user."""
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)

    # Acquire the shared dedup guard to prevent flooding the refinement queue.
    guard_key = f"refinement:pending:{user_id_str}"
    acquired = await redis.set(guard_key, "1", nx=True, ex=600)
    if not acquired:
        await message.answer(tr("refresh_memory.in_progress", locale), parse_mode=None)
        return

    try:
        await enqueue_refinement_job(redis, user_id_str, {
            "trigger": "manual_compact",
            "created_at": datetime.now(tz=UTC).isoformat(),
        })
    except Exception:  # noqa: BLE001
        try:
            await redis.delete(guard_key)
        except Exception:  # noqa: BLE001
            log.warning("compact_guard_cleanup_failed", internal_user_id=user_id_str)
        await message.answer(tr("refresh_memory.enqueue_failed", locale), parse_mode=None)
        log.warning("memory_compact_enqueue_failed", internal_user_id=user_id_str)
        return

    await message.answer(tr("refresh_memory.requested", locale), parse_mode=None)
    log.info("refresh_memory_requested", internal_user_id=user_id_str)


# Keep old name as alias for backwards compatibility
@router.message(Command("memory_compact_now"))
async def cmd_memory_compact_now(message: Message, db_user: User, redis: Redis) -> None:
    """Legacy alias for /refresh_memory."""
    await cmd_refresh_memory(message, db_user, redis)


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
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    try:
        async with _profile_write_lock(redis, db_session, db_user.id):
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
    except _ProfileLockBusyError:
        await message.answer(tr("profile.lock_in_progress", locale), parse_mode=None)


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
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    try:
        rolled_back = await rollback_to_previous(
            snapshot_store, db_user.id, session=db_session,
        )
    except RollbackError as exc:
        err_msg = str(exc)
        key = "rollback.no_snapshot" if "No active snapshot" in err_msg else "rollback.no_previous"
        await message.answer(tr(key, locale), parse_mode=None)
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


_DELETE_CONFIRM_PREFIX = "delete_confirm"
_DELETE_CONFIRM_TTL = 60


@router.message(Command("delete_my_data"))
async def cmd_delete_my_data(
    message: Message,
    db_user: User,
    redis: Redis,
) -> None:
    """Show a confirmation prompt before deleting all personal data."""
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)

    # Set a short-lived Redis guard so the callback knows the request is fresh.
    guard_key = f"{_DELETE_CONFIRM_PREFIX}:{user_id_str}"
    await redis.set(guard_key, "1", ex=_DELETE_CONFIRM_TTL)

    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text=tr("btn.yes", locale), callback_data="delete_data:yes",
        ),
        InlineKeyboardButton(
            text=tr("btn.no", locale), callback_data="delete_data:no",
        ),
    ]])
    await message.answer(
        tr("delete_my_data.confirm_prompt", locale),
        reply_markup=keyboard,
        parse_mode=None,
    )
    log.info("delete_my_data_prompt_shown", internal_user_id=user_id_str)


@router.callback_query(F.data == "delete_data:yes")
async def cb_delete_data_yes(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
    redis: Redis,
    snapshot_store: SnapshotStore,
) -> None:
    """Handle delete-data confirmation 'Yes'."""
    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)
    guard_key = f"{_DELETE_CONFIRM_PREFIX}:{user_id_str}"

    # Check the Redis guard (TTL-based).
    guard = await redis.getdel(guard_key)
    if guard is None:
        await callback.answer(tr("delete_my_data.expired", locale), show_alert=True)
        return

    await hard_delete_user(
        db_user.id, db_session, redis,
        telegram_user_id=db_user.telegram_user_id,
    )
    await snapshot_store.delete_for_user(db_user.id)
    log.info("delete_my_data_completed", internal_user_id=user_id_str)

    await callback.answer()
    try:
        if callback.message:
            await callback.message.edit_text(
                tr("delete_my_data.done", locale), reply_markup=None,
            )
    except Exception:  # noqa: BLE001
        log.warning("delete_my_data_edit_failed", internal_user_id=user_id_str)


@router.callback_query(F.data == "delete_data:no")
async def cb_delete_data_no(
    callback: CallbackQuery,
    db_user: User,
    redis: Redis,
) -> None:
    """Handle delete-data confirmation 'No'."""
    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)

    # Clear the guard if still present.
    guard_key = f"{_DELETE_CONFIRM_PREFIX}:{user_id_str}"
    await redis.delete(guard_key)

    await callback.answer()
    try:
        if callback.message:
            await callback.message.edit_text(
                tr("delete_my_data.cancelled", locale), reply_markup=None,
            )
    except Exception:  # noqa: BLE001
        log.warning("delete_my_data_cancel_edit_failed", internal_user_id=user_id_str)
    log.info("delete_my_data_cancelled", internal_user_id=user_id_str)


# --------------------------------------------------------------------------- #
# /reset — wipe child data and restart onboarding
# --------------------------------------------------------------------------- #


_RESET_CONFIRM_PREFIX = "reset_confirm"
_RESET_CONFIRM_TTL = 120


@router.message(Command("reset"))
async def cmd_reset(
    message: Message,
    db_user: User,
    redis: Redis,
) -> None:
    """Show a confirmation prompt before wiping child data and restarting onboarding."""
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)

    guard_key = f"{_RESET_CONFIRM_PREFIX}:{user_id_str}"
    await redis.set(guard_key, "1", ex=_RESET_CONFIRM_TTL)

    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text=tr("btn.yes", locale), callback_data="reset:yes",
        ),
        InlineKeyboardButton(
            text=tr("btn.no", locale), callback_data="reset:no",
        ),
    ]])
    await message.answer(
        tr("reset.confirm", locale),
        reply_markup=keyboard,
        parse_mode=None,
    )
    log.info("reset_prompt_shown", internal_user_id=user_id_str)


@router.callback_query(F.data == "reset:yes")
async def cb_reset_yes(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
    redis: Redis,
    snapshot_store: SnapshotStore,
) -> None:
    """Handle reset confirmation 'Yes': wipe child data and restart onboarding."""
    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)
    guard_key = f"{_RESET_CONFIRM_PREFIX}:{user_id_str}"

    guard = await redis.getdel(guard_key)
    if guard is None:
        await callback.answer(tr("reset.expired", locale), show_alert=True)
        return

    # Delete child tables in FK-safe order (leaves User row intact).
    for model in (
        MemoryCompaction,
        Job,
        BehaviorChangeEvent,
        ConversationMessage,
        PromptSnapshot,
        UserProfile,
    ):
        await db_session.execute(
            delete(model).where(model.user_id == db_user.id),  # type: ignore[attr-defined]
        )
    await db_session.flush()

    # Clear Redis keys scoped to this user.
    from companion_bot_core.privacy.delete_user import (
        _REDIS_KEY_PREFIXES,
        _REDIS_TELEGRAM_KEY_PREFIXES,
    )

    redis_keys = [f"{p}:{user_id_str}" for p in _REDIS_KEY_PREFIXES]
    redis_keys += [
        f"{p}:{db_user.telegram_user_id}" for p in _REDIS_TELEGRAM_KEY_PREFIXES
    ]
    redis_keys.append(f"{_ONBOARDING_PREFIX}:{user_id_str}")
    await redis.delete(*redis_keys)

    # Also clear snapshot store Redis pointers.
    await snapshot_store.delete_for_user(db_user.id)

    # Edit the confirmation message to "done".
    await callback.answer()
    try:
        if callback.message:
            await callback.message.edit_text(
                tr("reset.done", locale), reply_markup=None,
            )
    except Exception:  # noqa: BLE001
        log.warning("reset_edit_failed", internal_user_id=user_id_str)

    # Start onboarding.
    state_key = f"{_ONBOARDING_PREFIX}:{db_user.id}"
    await redis.set(
        state_key, json.dumps({"step": "name"}), ex=_ONBOARDING_TTL,
    )
    try:
        if callback.message:
            await callback.message.answer(
                tr("onboarding.step1_name", locale), parse_mode=None,
            )
    except Exception:  # noqa: BLE001
        log.warning("reset_onboarding_send_failed", internal_user_id=user_id_str)

    log.info("reset_completed", internal_user_id=user_id_str)


@router.callback_query(F.data == "reset:no")
async def cb_reset_no(
    callback: CallbackQuery,
    db_user: User,
    redis: Redis,
) -> None:
    """Handle reset confirmation 'No'."""
    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)

    guard_key = f"{_RESET_CONFIRM_PREFIX}:{user_id_str}"
    await redis.delete(guard_key)

    await callback.answer()
    try:
        if callback.message:
            await callback.message.edit_text(
                tr("reset.cancelled", locale), reply_markup=None,
            )
    except Exception:  # noqa: BLE001
        log.warning("reset_cancel_edit_failed", internal_user_id=user_id_str)
    log.info("reset_cancelled", internal_user_id=user_id_str)


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
    _is_group = _is_group_chat(message)
    _reply_fn = message.reply if _is_group else message.answer

    # --- Group chat filter: only respond to @mentions and replies to the bot ---
    if _is_group:
        is_reply_to_bot = (
            message.reply_to_message is not None
            and message.reply_to_message.from_user is not None
            and message.reply_to_message.from_user.is_bot
            and message.bot is not None
            and message.reply_to_message.from_user.id == message.bot.id
        )
        is_mention = False
        if message.entities and message.bot is not None:
            bot_username = (await message.bot.me()).username
            if bot_username:
                for entity in message.entities:
                    if entity.type == "mention":
                        mention_text = text[entity.offset:entity.offset + entity.length]
                        if mention_text.lower() == f"@{bot_username.lower()}":
                            is_mention = True
                            break
        if not is_reply_to_bot and not is_mention:
            return

    # --- Onboarding guard: intercept text during onboarding ---
    _ob_key = f"{_ONBOARDING_PREFIX}:{db_user.id}"
    _ob_raw = await redis.get(_ob_key)
    if _ob_raw is not None:
        # If the user sends a /command during onboarding, cancel onboarding
        # and let the message fall through to normal processing.
        if text.startswith("/"):
            await redis.delete(_ob_key)
            log.info("onboarding_escaped_by_command", user_id=user_id_str)
        else:
            _ob_state = json.loads(_ob_raw)
            _ob_step = _ob_state.get("step", "")

            if _ob_step == "name":
                # Process name input
                name = _sanitize_onboarding_name(text)
                if name is None:
                    await message.answer(
                        tr("onboarding.name_invalid", locale), parse_mode=None,
                    )
                    return
                # Guardrail check on onboarding name
                for _gcheck in (check_prompt_injection, check_unsafe_role_change):
                    if not _gcheck(name).allowed:
                        await message.answer(
                            tr("guardrail.command_blocked", locale), parse_mode=None,
                        )
                        return
                _ob_state["name"] = name
                _ob_state["step"] = "interests"
                await redis.set(
                    _ob_key, json.dumps(_ob_state), ex=_ONBOARDING_TTL,
                )
                await message.answer(
                    tr("onboarding.step2_interests", locale, name=html.escape(name)),
                    parse_mode=None,
                    reply_markup=_interest_keyboard(locale),
                )
                return

            # User is on a button-selection step — ask them to use buttons
            await message.answer(
                tr("onboarding.please_use_buttons", locale), parse_mode=None,
            )
            return

    ingress_start = time.perf_counter()
    reply = ""

    # Show typing indicator immediately so the user sees feedback while
    # the orchestrator loads context and waits for the first model token.
    try:
        await message.answer_chat_action(ChatAction.TYPING)
    except Exception:  # noqa: BLE001
        log.debug("typing_action_failed", internal_user_id=user_id_str)

    # Streaming state — populated by _on_chunk closure during inference.
    _placeholder: Message | None = None

    # Re-send typing indicator every 4 seconds until the streaming
    # placeholder is sent.  Telegram typing indicators expire after ~5s,
    # so this keeps the "typing..." visible during long context loads.
    async def _keep_typing() -> None:
        while _placeholder is None:
            await asyncio.sleep(4)
            if _placeholder is not None:
                break
            try:
                await message.answer_chat_action(ChatAction.TYPING)
            except Exception:  # noqa: BLE001
                break

    _typing_task = asyncio.create_task(_keep_typing())
    _stream_buf: list[str] = []
    _last_edit: float = 0.0
    _edit_interval: float = 0.3  # minimum seconds between edits
    _first_chunk_min_len: int = 4  # buffer before sending placeholder
    _pending_edit: asyncio.Task[None] | None = None

    async def _flush_edit() -> None:
        """Send or edit the streaming placeholder (runs as a background task)."""
        nonlocal _placeholder, _last_edit
        display = "".join(_stream_buf)[:_TG_MSG_LIMIT]
        if _placeholder is None:
            try:
                _placeholder = await _reply_fn(
                    display, parse_mode=None,
                )
                _last_edit = time.perf_counter()
            except Exception:  # noqa: BLE001
                log.warning(
                    "stream_placeholder_send_failed",
                    internal_user_id=user_id_str,
                )
        else:
            try:
                await _placeholder.edit_text(display, parse_mode=None)
                _last_edit = time.perf_counter()
            except Exception:  # noqa: BLE001
                log.debug(
                    "stream_edit_failed",
                    internal_user_id=user_id_str,
                )

    async def _on_chunk(chunk: str) -> None:
        nonlocal _pending_edit
        _stream_buf.append(chunk)
        now = time.perf_counter()

        # Decide whether we should flush an edit to Telegram.
        should_flush = (
            (
                _placeholder is None
                and len("".join(_stream_buf)) >= _first_chunk_min_len
            )
            or (
                _placeholder is not None
                and now - _last_edit >= _edit_interval
            )
        )

        if not should_flush:
            return

        # Only fire a new edit task if the previous one has finished,
        # so the SSE read loop is never blocked by Telegram RTT.
        if _pending_edit is not None and not _pending_edit.done():
            return

        _pending_edit = asyncio.create_task(_flush_edit())

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
                max_tokens=settings.chat_max_tokens,
                encryptor=encryptor,
                locale=locale,
                on_stream_chunk=_on_chunk,
                context_message_limit=settings.context_message_limit,
                feedback_session_interval=settings.feedback_session_interval,
                feedback_cooldown_days=settings.feedback_cooldown_days,
            )
        except Exception:
            _typing_task.cancel()
            log.exception(
                "process_message_failed",
                internal_user_id=user_id_str,
            )
            # Send a user-facing error reply before re-raising so the
            # middleware rolls back the DB transaction (prevents committing
            # partially-flushed state such as assistant messages the user
            # never saw).
            try:
                await _reply_fn(
                    tr("handle.error", locale),
                    parse_mode=None,
                )
            except Exception:  # noqa: BLE001
                log.warning("error_reply_send_failed", internal_user_id=user_id_str)
            raise
        finally:
            _typing_task.cancel()

        # Silently consume the refinement notice — profile updates are
        # applied without user-facing notifications.  The user can always
        # check their profile via /memory.
        try:
            await check_and_clear_user_notice(redis, user_id_str)
        except Exception:  # noqa: BLE001
            log.warning("notice_check_failed", internal_user_id=user_id_str)

        # Check if a pending confirmation was just created — if so, attach
        # inline Yes/No buttons so the user doesn't have to type "yes"/"no".
        from companion_bot_core.orchestrator.dialogue_state import (
            get_pending_change as _get_pending_for_buttons,
        )

        pending_for_buttons = await _get_pending_for_buttons(
            redis, user_id_str,
        )
        confirm_kb: InlineKeyboardMarkup | None = None
        if pending_for_buttons is not None:
            confirm_kb = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(
                    text=tr("btn.yes", locale),
                    callback_data="confirm:yes",
                ),
                InlineKeyboardButton(
                    text=tr("btn.no", locale),
                    callback_data="confirm:no",
                ),
            ]])

        # Deliver the final reply text.  If a placeholder was sent during
        # streaming, edit it with the final text; otherwise use message.answer().
        # First, wait for any in-flight streaming edit to finish.
        if _pending_edit is not None and not _pending_edit.done():
            try:
                await _pending_edit
            except Exception:  # noqa: BLE001
                log.debug(
                    "stream_pending_edit_await_failed",
                    internal_user_id=user_id_str,
                )

        try:
            chunks = [
                reply[i : i + _TG_MSG_LIMIT]
                for i in range(0, len(reply), _TG_MSG_LIMIT)
            ]
            if _placeholder is not None:
                # Always do the final edit — streaming edits are rate-limited
                # so the Telegram message may lag behind the in-memory buffer.
                first_kb = confirm_kb if len(chunks) == 1 else None
                try:
                    await _placeholder.edit_text(
                        chunks[0], parse_mode=None, reply_markup=first_kb,
                    )
                except Exception:  # noqa: BLE001
                    log.debug(
                        "stream_final_edit_failed",
                        internal_user_id=user_id_str,
                    )
                # Send overflow chunks as new messages.
                for idx, chunk in enumerate(chunks[1:], start=1):
                    kb = confirm_kb if idx == len(chunks) - 1 else None
                    await _reply_fn(
                        chunk, parse_mode=None, reply_markup=kb,
                    )
            else:
                # No streaming placeholder \u2014 early-exit paths.
                for idx, chunk in enumerate(chunks):
                    kb = confirm_kb if idx == len(chunks) - 1 else None
                    await _reply_fn(
                        chunk, parse_mode=None, reply_markup=kb,
                    )
        except Exception as exc:
            log.warning(
                "reply_send_failed",
                internal_user_id=user_id_str,
                reply_length=len(reply),
                error=str(exc),
            )

    elapsed_ms = round((time.perf_counter() - ingress_start) * 1000, 2)
    log.info(
        "message_handled",
        internal_user_id=user_id_str,
        reply_length=len(reply),
        elapsed_ms=elapsed_ms,
    )


# --------------------------------------------------------------------------- #
# /settings — inline keyboard menu
# --------------------------------------------------------------------------- #


def _settings_keyboard(locale: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=tr("btn.tone", locale), callback_data="settings:tone"),
            InlineKeyboardButton(text=tr("btn.persona", locale), callback_data="settings:persona"),
        ],
        [
            InlineKeyboardButton(text=tr("btn.skills", locale), callback_data="settings:skills"),
            InlineKeyboardButton(
                text=tr("btn.language", locale),
                callback_data="settings:language",
            ),
        ],
    ])


@router.message(Command("settings"))
async def cmd_settings(message: Message, db_user: User) -> None:
    """Show the settings menu with inline buttons."""
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    await message.answer(
        tr("settings.choose", locale),
        reply_markup=_settings_keyboard(locale),
        parse_mode=None,
    )
    log.info("cmd_settings", internal_user_id=str(db_user.id))


@router.callback_query(F.data == "settings:back")
async def cb_settings_back(callback: CallbackQuery, db_user: User) -> None:
    """Return to the main /settings menu."""
    locale = _user_locale(db_user)
    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            tr("settings.choose", locale),
            reply_markup=_settings_keyboard(locale),
        )
    await callback.answer()


# --------------------------------------------------------------------------- #
# Tone picker (inline keyboard)
# --------------------------------------------------------------------------- #


_TONE_LIST = sorted(VALID_TONES)


def _tone_keyboard(locale: str) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text=tr(f"tone_label.{tone}", locale), callback_data=f"tone:{tone}")]
        for tone in _TONE_LIST
    ]
    back = InlineKeyboardButton(text=tr("btn.back", locale), callback_data="settings:back")
    buttons.append([back])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@router.callback_query(F.data == "settings:tone")
async def cb_settings_tone(callback: CallbackQuery, db_user: User) -> None:
    locale = _user_locale(db_user)
    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            tr("tone.pick", locale),
            reply_markup=_tone_keyboard(locale),
        )
    await callback.answer()


@router.callback_query(F.data.startswith("tone:"))
async def cb_tone_pick(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    encryptor: FieldEncryptor | None = None,
    redis: Redis | None = None,
) -> None:
    tone = (callback.data or "").split(":", 1)[1]
    locale = _user_locale(db_user)
    enc = encryptor or NOOP_ENCRYPTOR

    if tone not in VALID_TONES:
        await callback.answer()
        return

    try:
        async with _profile_write_lock(redis, db_session, db_user.id):
            await acquire_profile_advisory_lock(db_session, db_user.id)
            profile = await get_or_create_profile(db_session, db_user.id)
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
            if callback.message is not None:
                await callback.message.edit_text(  # type: ignore[union-attr]
                    tr("tone.set", locale, tone=tone),
                )
            await callback.answer()
            log.info("cb_tone_pick", internal_user_id=str(db_user.id), tone=tone)
    except _ProfileLockBusyError:
        await callback.answer(tr("profile.lock_in_progress", locale))


# --------------------------------------------------------------------------- #
# Settings sub-menus
# --------------------------------------------------------------------------- #


@router.callback_query(F.data == "settings:language")
async def cb_settings_language(callback: CallbackQuery, db_user: User) -> None:
    locale = _user_locale(db_user)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Русский", callback_data="lang:ru"),
            InlineKeyboardButton(text="English", callback_data="lang:en"),
        ],
        [InlineKeyboardButton(text=tr("btn.back", locale), callback_data="settings:back")],
    ])
    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            tr("set_language.help", locale),
            reply_markup=kb,
        )
    await callback.answer()


@router.callback_query(F.data.startswith("lang:"))
async def cb_lang_pick(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
) -> None:
    new_locale = (callback.data or "").split(":", 1)[1]
    if new_locale not in SUPPORTED_LOCALES:
        await callback.answer()
        return
    db_user.locale = new_locale
    await db_session.flush()
    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            tr("set_language.updated", new_locale),
        )
    await callback.answer()
    log.info("cb_lang_pick", internal_user_id=str(db_user.id), locale=new_locale)


# --------------------------------------------------------------------------- #
# Persona browser (inline keyboard)  — /personas
# --------------------------------------------------------------------------- #

# Deep persona definitions with personalities
DEEP_PERSONAS: dict[str, dict[str, str]] = {
    "study_buddy": {
        "name_ru": "Учебный друг",
        "name_en": "Study Buddy",
        "desc_ru": (
            "Помогаю разбираться в сложных темах. Использую метод Сократа, "
            "разбиваю сложное на простые шаги и помогаю запоминать через повторение."
        ),
        "desc_en": (
            "I help you understand complex topics. I use the Socratic method, "
            "break down complex ideas into simple steps, and help with spaced repetition."
        ),
        "persona_text": (
            "You are a patient, encouraging study companion. "
            "Use the Socratic method: ask guiding questions rather than giving answers directly. "
            "Break complex concepts into digestible steps. Suggest mnemonic devices and "
            "spaced repetition when appropriate. Celebrate small wins in understanding."
        ),
    },
    "creative_muse": {
        "name_ru": "Творческая муза",
        "name_en": "Creative Muse",
        "desc_ru": (
            "Вдохновляю и помогаю с творческими проектами. "
            "Генерирую идеи, даю обратную связь и помогаю преодолевать творческий блок."
        ),
        "desc_en": (
            "I inspire and help with creative projects. "
            "I generate ideas, give feedback, and help overcome creative blocks."
        ),
        "persona_text": (
            "You are an enthusiastic creative collaborator. "
            "Generate unexpected ideas and connections. Give constructive, honest feedback "
            "on creative work while staying encouraging. When the user is stuck, "
            "suggest exercises and prompts to unblock creativity. Reference artistic "
            "movements and techniques when relevant."
        ),
    },
    "life_coach": {
        "name_ru": "Лайф-коуч",
        "name_en": "Life Coach",
        "desc_ru": (
            "Помогаю ставить цели и двигаться к ним. "
            "Задаю правильные вопросы, помогаю расставить приоритеты и поддерживаю мотивацию."
        ),
        "desc_en": (
            "I help you set goals and work toward them. "
            "I ask the right questions, help prioritize, and keep you motivated."
        ),
        "persona_text": (
            "You are a warm but direct life coach. "
            "Help the user clarify their goals through thoughtful questions. "
            "Break big goals into actionable steps. Gently challenge assumptions "
            "and limiting beliefs. Celebrate progress. Never be preachy — "
            "be a supportive thinking partner."
        ),
    },
    "tech_mentor": {
        "name_ru": "Тех-ментор",
        "name_en": "Tech Mentor",
        "desc_ru": (
            "Помогаю с программированием и технологиями. "
            "Объясняю архитектурные решения, ревьюю код и подсказываю лучшие практики."
        ),
        "desc_en": (
            "I help with programming and technology. "
            "I explain architectural decisions, review code, and suggest best practices."
        ),
        "persona_text": (
            "You are an experienced senior developer and mentor. "
            "Explain architectural trade-offs clearly. When reviewing code, "
            "focus on the most impactful improvements. Teach idiomatic patterns "
            "for the language being used. Ask clarifying questions before diving "
            "into solutions. Prefer simple, readable code over clever abstractions."
        ),
    },
}


def _personas_keyboard(locale: str) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    name_key = "name_ru" if locale == "ru" else "name_en"
    for key, persona in DEEP_PERSONAS.items():
        buttons.append([
            InlineKeyboardButton(text=persona[name_key], callback_data=f"persona_view:{key}"),
        ])
    # Add seed personas (skip those already shown as deep personas)
    for key in SEED_PERSONAS:
        if key in DEEP_PERSONAS:
            continue
        buttons.append([
            InlineKeyboardButton(
                text=tr(f"persona_label.{key}", locale),
                callback_data=f"persona_set:{key}",
            ),
        ])
    back = InlineKeyboardButton(text=tr("btn.back", locale), callback_data="settings:back")
    buttons.append([back])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@router.message(Command("personas"))
async def cmd_personas(message: Message, db_user: User) -> None:
    """Browse available personas."""
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    await message.answer(
        tr("personas.title", locale),
        reply_markup=_personas_keyboard(locale),
        parse_mode=None,
    )
    log.info("cmd_personas", internal_user_id=str(db_user.id))


@router.callback_query(F.data == "settings:persona")
async def cb_settings_persona(callback: CallbackQuery, db_user: User) -> None:
    locale = _user_locale(db_user)
    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            tr("personas.title", locale),
            reply_markup=_personas_keyboard(locale),
        )
    await callback.answer()


@router.callback_query(F.data.startswith("persona_view:"))
async def cb_persona_view(callback: CallbackQuery, db_user: User) -> None:
    locale = _user_locale(db_user)
    key = (callback.data or "").split(":", 1)[1]
    persona = DEEP_PERSONAS.get(key)
    if persona is None:
        await callback.answer()
        return
    name_key = "name_ru" if locale == "ru" else "name_en"
    desc_key = "desc_ru" if locale == "ru" else "desc_en"
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=tr("btn.personas.select", locale),
            callback_data=f"persona_deep:{key}",
        )],
        [InlineKeyboardButton(
            text=tr("btn.back", locale),
            callback_data="settings:persona",
        )],
    ])
    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            tr("personas.preview", locale, name=persona[name_key], description=persona[desc_key]),
            reply_markup=kb,
        )
    await callback.answer()


@router.callback_query(F.data.startswith("persona_deep:"))
async def cb_persona_deep_select(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    encryptor: FieldEncryptor | None = None,
    redis: Redis | None = None,
) -> None:
    locale = _user_locale(db_user)
    key = (callback.data or "").split(":", 1)[1]
    persona = DEEP_PERSONAS.get(key)
    if persona is None:
        await callback.answer()
        return

    enc = encryptor or NOOP_ENCRYPTOR
    name_key = "name_ru" if locale == "ru" else "name_en"
    persona_name = persona[name_key]

    try:
        async with _profile_write_lock(redis, db_session, db_user.id):
            await acquire_profile_advisory_lock(db_session, db_user.id)
            profile = await get_or_create_profile(db_session, db_user.id)
            raw_tone = (
                enc.decrypt_safe(profile.tone, default="") if profile.tone else None
            )
            profile.persona_name = enc.encrypt(persona_name)
            await db_session.flush()

            current = await snapshot_store.get_active(db_user.id)
            if current is not None:
                base_template = _extract_base_template(current.system_prompt)
                skill_packs_dict: dict[str, str] = {
                    k: str(v) for k, v in current.skill_prompts_json.items()
                }
                ltp = _extract_section(current.system_prompt, "Long-term Profile")
                raw_skills: dict[str, object] = dict(current.skill_prompts_json)
            else:
                base_template = _DEFAULT_SYSTEM_TEMPLATE
                skill_packs_dict = {}
                ltp = ""
                raw_skills = {}

            persona_segment = f"Имя пользователя: {persona_name}\n{persona['persona_text']}"
            if raw_tone:
                persona_segment += f"\nTone: {raw_tone}"
            components = _PromptComponents(
                base_system_template=base_template,
                persona_segment=persona_segment,
                skill_packs=skill_packs_dict,
                long_term_profile=ltp,
            )
            system_prompt = _build_system_prompt(components)
            version = await snapshot_store.next_version(db_user.id)
            record = _SnapshotRecord(
                user_id=db_user.id,
                version=version,
                system_prompt=system_prompt,
                skill_prompts_json=raw_skills,
                source="user_command",
            )
            await snapshot_store.save(record, session=db_session)
            await snapshot_store.set_active(db_user.id, record.id, session=db_session)

            if callback.message is not None:
                await callback.message.edit_text(  # type: ignore[union-attr]
                    tr("personas.selected", locale, name=persona_name),
                )
            await callback.answer()
            log.info("cb_persona_deep", internal_user_id=str(db_user.id), persona=key)
    except _ProfileLockBusyError:
        await callback.answer(tr("profile.lock_in_progress", locale))


@router.callback_query(F.data.startswith("persona_set:"))
async def cb_persona_set(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    encryptor: FieldEncryptor | None = None,
    redis: Redis | None = None,
) -> None:
    locale = _user_locale(db_user)
    key = (callback.data or "").split(":", 1)[1]
    if key not in SEED_PERSONAS:
        await callback.answer()
        return

    enc = encryptor or NOOP_ENCRYPTOR
    try:
        async with _profile_write_lock(redis, db_session, db_user.id):
            await acquire_profile_advisory_lock(db_session, db_user.id)
            profile = await get_or_create_profile(db_session, db_user.id)
            raw_tone = (
                enc.decrypt_safe(profile.tone, default="") if profile.tone else None
            )
            profile.persona_name = enc.encrypt(key)
            await db_session.flush()
            await rebuild_and_save_snapshot(
                snapshot_store, db_user.id, key, raw_tone,
                session=db_session,
            )
            if callback.message is not None:
                await callback.message.edit_text(  # type: ignore[union-attr]
                    tr("personas.selected", locale, name=key.capitalize()),
                )
            await callback.answer()
            log.info("cb_persona_set", internal_user_id=str(db_user.id), persona=key)
    except _ProfileLockBusyError:
        await callback.answer(tr("profile.lock_in_progress", locale))


# --------------------------------------------------------------------------- #
# Skills catalog — /skills
# --------------------------------------------------------------------------- #

# Pre-built skill catalog
SKILL_CATALOG: dict[str, dict[str, str]] = {
    "code_assistant": {
        "name_ru": "Код",
        "name_en": "Code",
        "prompt": (
            "When helping with code: always specify the language, "
            "prefer idiomatic patterns, and include brief comments for non-obvious logic."
        ),
    },
    "study_coach": {
        "name_ru": "Учёба",
        "name_en": "Study",
        "prompt": (
            "When asked about learning topics: use the Socratic method where helpful, "
            "suggest spaced-repetition techniques, and break complex concepts into steps."
        ),
    },
    "writing_helper": {
        "name_ru": "Письмо",
        "name_en": "Writing",
        "prompt": (
            "Help with writing: suggest structural improvements, catch grammar issues, "
            "and adapt tone to the audience. Offer alternatives rather than rewriting everything."
        ),
    },
    "language_tutor": {
        "name_ru": "Языки",
        "name_en": "Languages",
        "prompt": (
            "Act as a language tutor: correct mistakes gently, explain grammar rules "
            "when asked, and suggest natural phrasing. Practice conversation when appropriate."
        ),
    },
    "fitness_coach": {
        "name_ru": "Фитнес",
        "name_en": "Fitness",
        "prompt": (
            "Provide fitness and wellness guidance: suggest exercises appropriate to "
            "the user's level, explain proper form, and help build workout plans. "
            "Always recommend consulting a doctor for medical concerns."
        ),
    },
    "cooking_helper": {
        "name_ru": "Кулинария",
        "name_en": "Cooking",
        "prompt": (
            "Help with cooking: suggest recipes based on available ingredients, "
            "explain techniques, and offer substitutions. Adapt portions and difficulty "
            "to the user's experience level."
        ),
    },
}


def _skills_keyboard(
    locale: str,
    active_skills: dict[str, object],
) -> InlineKeyboardMarkup:
    name_key = "name_ru" if locale == "ru" else "name_en"
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for key, skill in SKILL_CATALOG.items():
        is_active = key in active_skills
        label = f"{'[x] ' if is_active else '[ ] '}{skill[name_key]}"
        row.append(InlineKeyboardButton(text=label, callback_data=f"skill_toggle:{key}"))
        if len(row) == 3:  # noqa: PLR2004
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    back = InlineKeyboardButton(text=tr("btn.back", locale), callback_data="settings:back")
    buttons.append([back])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@router.message(Command("skills"))
async def cmd_skills(
    message: Message,
    db_user: User,
    snapshot_store: SnapshotStore,
) -> None:
    """Show the skill catalog with toggle buttons."""
    if await _guard_private_only(message, db_user):
        return
    locale = _user_locale(db_user)
    snapshot = await snapshot_store.get_active(db_user.id)
    active_skills = snapshot.skill_prompts_json if snapshot else {}

    await message.answer(
        tr("skills.description", locale),
        reply_markup=_skills_keyboard(locale, active_skills),
        parse_mode=None,
    )
    log.info("cmd_skills", internal_user_id=str(db_user.id))


@router.callback_query(F.data == "settings:skills")
async def cb_settings_skills(
    callback: CallbackQuery,
    db_user: User,
    snapshot_store: SnapshotStore,
) -> None:
    locale = _user_locale(db_user)
    snapshot = await snapshot_store.get_active(db_user.id)
    active_skills = snapshot.skill_prompts_json if snapshot else {}
    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            tr("skills.description", locale),
            reply_markup=_skills_keyboard(locale, active_skills),
        )
    await callback.answer()


@router.callback_query(F.data.startswith("skill_toggle:"))
async def cb_skill_toggle(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    redis: Redis | None = None,
) -> None:
    locale = _user_locale(db_user)
    key = (callback.data or "").split(":", 1)[1]
    skill = SKILL_CATALOG.get(key)
    if skill is None:
        await callback.answer()
        return

    name_key = "name_ru" if locale == "ru" else "name_en"

    try:
        async with _profile_write_lock(redis, db_session, db_user.id):
            await acquire_profile_advisory_lock(db_session, db_user.id)
            snapshot = await snapshot_store.get_active(db_user.id)
            active_skills = dict(snapshot.skill_prompts_json) if snapshot else {}

            if snapshot is not None:
                base = _extract_base_template(snapshot.system_prompt)
                sp: dict[str, str] = {k: str(v) for k, v in snapshot.skill_prompts_json.items()}
                ltp = _extract_section(snapshot.system_prompt, "Long-term Profile")
                ps = _extract_section(snapshot.system_prompt, "Persona")
            else:
                base = _DEFAULT_SYSTEM_TEMPLATE
                sp = {}
                ltp = ""
                ps = ""

            if key in active_skills:
                sp.pop(key, None)
                msg = tr("skills.toggled_off", locale, name=skill[name_key])
            else:
                sp[key] = skill["prompt"]
                msg = tr("skills.toggled_on", locale, name=skill[name_key])

            comp = _PromptComponents(
                base_system_template=base,
                persona_segment=ps,
                skill_packs=sp,
                long_term_profile=ltp,
            )
            sys_p = _build_system_prompt(comp)
            ver = await snapshot_store.next_version(db_user.id)
            rec = _SnapshotRecord(
                user_id=db_user.id,
                version=ver,
                system_prompt=sys_p,
                skill_prompts_json=dict(sp),
                source="user_command",
            )
            await snapshot_store.save(rec, session=db_session)
            await snapshot_store.set_active(db_user.id, rec.id, session=db_session)

            # Refresh the keyboard with updated state
            snapshot = await snapshot_store.get_active(db_user.id)
            new_active = snapshot.skill_prompts_json if snapshot else {}
            if callback.message is not None:
                await callback.message.edit_text(  # type: ignore[union-attr]
                    msg,
                    reply_markup=_skills_keyboard(locale, new_active),
                )
            await callback.answer()
            log.info("cb_skill_toggle", internal_user_id=str(db_user.id), skill=key)
    except _ProfileLockBusyError:
        await callback.answer(tr("profile.lock_in_progress", locale))


# --------------------------------------------------------------------------- #
# Confirmation inline buttons (Yes / No)
# --------------------------------------------------------------------------- #


@router.callback_query(F.data == "confirm:yes")
async def cb_confirm_yes(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    redis: Redis,
    encryptor: FieldEncryptor | None = None,
) -> None:
    """Handle confirmation 'Yes' button press."""
    from companion_bot_core.orchestrator.dialogue_state import (
        clear_pending_change as _clear_pending,
    )
    from companion_bot_core.orchestrator.dialogue_state import (
        get_pending_change as _get_pending,
    )
    from companion_bot_core.orchestrator.orchestrator import (
        _apply_behavior_change as _apply_change,
    )
    from companion_bot_core.orchestrator.orchestrator import (
        _record_behavior_event as _record_event,
    )

    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)
    pending = await _get_pending(redis, user_id_str)

    if pending is None:
        await callback.answer(tr("confirm.expired", locale), show_alert=True)
        return

    try:
        async with _profile_write_lock(redis, db_session, db_user.id):
            applied = await _apply_change(
                intent=pending.detection_result.intent,
                message_text=pending.original_message,
                user_id=db_user.id,
                session=db_session,
                snapshot_store=snapshot_store,
                encryptor=encryptor,
            )
            await _record_event(
                db_session,
                db_user.id,
                pending.detection_result,
                applied=applied,
                confirmed=True,
            )
            await _clear_pending(redis, user_id_str)

            if callback.message is not None:
                text = (
                    tr("orchestrator.change_applied", locale)
                    if applied
                    else tr("orchestrator.change_apply_failed", locale)
                )
                await callback.message.edit_text(  # type: ignore[union-attr]
                    text,
                )
            await callback.answer()
            log.info("cb_confirm_yes", internal_user_id=user_id_str, applied=applied)
    except _ProfileLockBusyError:
        await callback.answer(tr("profile.lock_in_progress", locale))


@router.callback_query(F.data == "confirm:no")
async def cb_confirm_no(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
    redis: Redis,
) -> None:
    """Handle confirmation 'No' button press."""
    from companion_bot_core.orchestrator.dialogue_state import (
        clear_pending_change as _clear_pending,
    )
    from companion_bot_core.orchestrator.dialogue_state import (
        get_pending_change as _get_pending,
    )
    from companion_bot_core.orchestrator.orchestrator import (
        _record_behavior_event as _record_event,
    )

    locale = _user_locale(db_user)
    user_id_str = str(db_user.id)
    pending = await _get_pending(redis, user_id_str)
    if pending is None:
        await callback.answer(tr("confirm.expired", locale), show_alert=True)
        return
    await _record_event(
        db_session,
        db_user.id,
        pending.detection_result,
        applied=False,
        confirmed=False,
    )
    await _clear_pending(redis, user_id_str)

    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            tr("orchestrator.change_cancelled", locale),
        )
    await callback.answer()
    log.info("cb_confirm_no", internal_user_id=user_id_str)


# --------------------------------------------------------------------------- #
# Onboarding flow (callback queries)
# --------------------------------------------------------------------------- #


_ONBOARDING_PREFIX = "onboarding"
_ONBOARDING_TTL = 600  # 10 minutes


@router.callback_query(F.data.startswith("onboard_interest:"))
async def cb_onboard_interest(
    callback: CallbackQuery,
    db_user: User,
    redis: Redis,
) -> None:
    """Handle interest selection in onboarding step 2."""
    locale = _user_locale(db_user)
    interest = (callback.data or "").split(":", 1)[1]
    state_key = f"{_ONBOARDING_PREFIX}:{db_user.id}"

    # Store selected interest (skip means no interest)
    raw = await redis.get(state_key)
    state = json.loads(raw) if raw else {}
    if interest != "skip":
        state["interest"] = interest
    state["step"] = "tone"
    await redis.set(state_key, json.dumps(state), ex=_ONBOARDING_TTL)

    # Show tone selection (step 3)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text=tr("tone_label.friendly", locale),
                callback_data="onboard_tone:friendly",
            ),
            InlineKeyboardButton(
                text=tr("tone_label.professional", locale),
                callback_data="onboard_tone:professional",
            ),
        ],
        [
            InlineKeyboardButton(
                text=tr("tone_label.playful", locale),
                callback_data="onboard_tone:playful",
            ),
            InlineKeyboardButton(
                text=tr("tone_label.concise", locale),
                callback_data="onboard_tone:concise",
            ),
        ],
        [
            InlineKeyboardButton(
                text=tr("onboarding.skip", locale),
                callback_data="onboard_tone:skip",
            ),
        ],
    ])
    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            tr("onboarding.step3_tone", locale),
            reply_markup=kb,
        )
    await callback.answer()


@router.callback_query(F.data.startswith("onboard_tone:"))
async def cb_onboard_tone(
    callback: CallbackQuery,
    db_user: User,
    db_session: AsyncSession,
    snapshot_store: SnapshotStore,
    redis: Redis,
    encryptor: FieldEncryptor | None = None,
) -> None:
    """Handle tone selection in onboarding step 3 — finalize onboarding."""
    locale = _user_locale(db_user)
    tone = (callback.data or "").split(":", 1)[1]
    enc = encryptor or NOOP_ENCRYPTOR
    state_key = f"{_ONBOARDING_PREFIX}:{db_user.id}"

    raw = await redis.get(state_key)
    state = json.loads(raw) if raw else {}
    interest = state.get("interest", "")
    name = state.get("name", "")

    try:
        async with _profile_write_lock(redis, db_session, db_user.id):
            await acquire_profile_advisory_lock(db_session, db_user.id)
            profile = await get_or_create_profile(db_session, db_user.id)
            if name:
                profile.persona_name = enc.encrypt(name)
            if tone != "skip":
                profile.tone = enc.encrypt(tone)
            await db_session.flush()

            effective_tone = tone if tone != "skip" else None
            # Build the complete initial snapshot in one step.  Calling
            # rebuild_and_save_snapshot followed by add_fact_to_profile
            # caused a visibility bug: add_fact_to_profile's get_active()
            # opened a separate DB session that couldn't see the uncommitted
            # snapshot from rebuild_and_save_snapshot, so the interest-fact
            # snapshot overwrote the name+tone snapshot with an empty
            # persona segment.  Building everything atomically avoids this.
            from companion_bot_core.prompt.helpers import build_persona_segment

            long_term = ""
            if interest:
                fact = tr(f"interest_fact.{interest}", locale)
                long_term = f"[manual] {fact}"

            components = _PromptComponents(
                base_system_template=_DEFAULT_SYSTEM_TEMPLATE,
                persona_segment=build_persona_segment(name or None, effective_tone),
                skill_packs={},
                long_term_profile=long_term,
            )
            system_prompt = _build_system_prompt(components)
            version = await snapshot_store.next_version(db_user.id)
            record = _SnapshotRecord(
                user_id=db_user.id,
                version=version,
                system_prompt=system_prompt,
                skill_prompts_json={},
                source="user_command",
            )
            await snapshot_store.save(record, session=db_session)
            await snapshot_store.set_active(db_user.id, record.id, session=db_session)

            await redis.delete(state_key)

            if callback.message is not None:
                await callback.message.edit_text(  # type: ignore[union-attr]
                    tr("onboarding.done", locale),
                )
            await callback.answer()
            log.info("onboarding_completed", internal_user_id=str(db_user.id))
    except _ProfileLockBusyError:
        await callback.answer(tr("profile.lock_in_progress", locale))


# --------------------------------------------------------------------------- #
# Unsupported content-type fallback handlers
# --------------------------------------------------------------------------- #
# Registered AFTER all other @router.message handlers so that commands and
# F.text are matched first.  The catch-all @router.message() MUST be last.


@router.message(F.photo)
async def handle_photo(message: Message, db_user: User) -> None:
    if _is_group_chat(message):
        return
    await message.answer(tr("unsupported.photo", _user_locale(db_user)), parse_mode=None)


@router.message(F.voice)
async def handle_voice(message: Message, db_user: User) -> None:
    if _is_group_chat(message):
        return
    await message.answer(tr("unsupported.voice", _user_locale(db_user)), parse_mode=None)


@router.message(F.sticker)
async def handle_sticker(message: Message, db_user: User) -> None:
    if _is_group_chat(message):
        return
    await message.answer(tr("unsupported.sticker", _user_locale(db_user)), parse_mode=None)


@router.message(F.document)
async def handle_document(message: Message, db_user: User) -> None:
    if _is_group_chat(message):
        return
    await message.answer(tr("unsupported.document", _user_locale(db_user)), parse_mode=None)


@router.message()
async def handle_unsupported(message: Message, db_user: User) -> None:
    """Catch-all for any message type not handled above."""
    if _is_group_chat(message):
        return
    await message.answer(tr("unsupported.other", _user_locale(db_user)), parse_mode=None)
