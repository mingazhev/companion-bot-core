"""Context assembly for the conversation orchestrator.

Provides helpers to load recent conversation history from PostgreSQL and to
build the :class:`~companion_bot_core.inference.schemas.UserContext` object that is passed
to the inference adapter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from sqlalchemy import select

from companion_bot_core.db.models import ConversationMessage
from companion_bot_core.i18n import normalize_locale, tr
from companion_bot_core.inference.schemas import ChatMessage, UserContext
from companion_bot_core.logging_config import get_logger
from companion_bot_core.privacy.field_encryption import NOOP_ENCRYPTOR, FieldEncryptor
from companion_bot_core.proactive.warm_return import build_warm_return_hint
from companion_bot_core.prompt.merge_builder import build_system_prompt, extract_section
from companion_bot_core.prompt.schemas import (
    DEFAULT_SYSTEM_TEMPLATE,
    PromptComponents,
    SnapshotRecord,
)
from companion_bot_core.redis.prompt_cache import cache_prompt, get_cached_prompt

if TYPE_CHECKING:
    from uuid import UUID

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from companion_bot_core.prompt.snapshot_store import SnapshotStore

log = get_logger(__name__)

_DEFAULT_MAX_TOKENS = 2048
# Type alias for valid chat message roles — used to satisfy ChatMessage.role narrowing
_MessageRole = Literal["system", "user", "assistant"]

# Redis key prefix for tracking user last activity timestamp
_LAST_ACTIVE_PREFIX = "last_active"
# Gap (in seconds) before injecting a continuity hint
_CONTINUITY_GAP_SECONDS = 3600  # 1 hour
# Redis key prefix for suggestion cooldown
_SUGGESTION_COOLDOWN_PREFIX = "suggestion:last"
_SUGGESTION_COOLDOWN_TTL = 86400  # 24 hours
# Gap (in seconds) before considering a proactive suggestion
_SUGGESTION_GAP_SECONDS = 14400  # 4 hours
# Fewer than this many messages means the user just finished onboarding
_FIRST_CONTACT_THRESHOLD = 3


async def load_recent_messages(
    session: AsyncSession,
    user_id: UUID,
    limit: int = 20,
    encryptor: FieldEncryptor | None = None,
) -> list[ChatMessage]:
    """Return the most recent non-expired conversation messages for *user_id*.

    Messages are ordered **oldest first** so they can be fed directly into the
    message list as conversation history.

    Args:
        session:    Active database session.
        user_id:    Internal (UUID) user identifier.
        limit:      Maximum number of messages to load (default 20).
        encryptor:  Optional field encryptor.  Decrypts ``content`` from DB rows.
                    ``None`` uses a disabled (pass-through) encryptor so legacy
                    unencrypted rows are returned unchanged.

    Returns:
        List of :class:`~companion_bot_core.inference.schemas.ChatMessage` in chronological
        order (oldest first).
    """
    enc = encryptor or NOOP_ENCRYPTOR
    now = datetime.now(tz=UTC)
    stmt = (
        select(ConversationMessage)
        .where(ConversationMessage.user_id == user_id)
        .where(
            (ConversationMessage.ttl_expires_at.is_(None))
            | (ConversationMessage.ttl_expires_at > now)
        )
        .order_by(ConversationMessage.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())

    # Reverse to chronological (oldest first) for the inference message list.
    # Use decrypt_safe so legacy unencrypted rows are returned as-is.
    return [
        ChatMessage(
            role=cast("_MessageRole", row.role),
            content=enc.decrypt_safe(row.content, default=row.content),
        )
        for row in reversed(rows)
    ]


def _extract_recent_topics(messages: list[ChatMessage], max_topics: int = 3) -> list[str]:
    """Extract recent conversation topics from message history.

    Looks at user messages and extracts the first meaningful sentence
    (up to 80 chars) from each unique message.
    """
    topics: list[str] = []
    seen: set[str] = set()
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        text = msg.content.strip()
        if not text or len(text) < 10:  # noqa: PLR2004
            continue
        # Find the first sentence longer than 10 chars.
        sentences = text.replace("?", ".").replace("!", ".").split(".")
        topic = ""
        for sentence in sentences:
            s = sentence.strip()
            if len(s) > 10:  # noqa: PLR2004
                topic = s[:80].strip()
                break
        if not topic:
            topic = text[:80].strip()
        topic_lower = topic.lower()
        if topic_lower not in seen and len(topic) > 5:  # noqa: PLR2004
            topics.append(topic)
            seen.add(topic_lower)
        if len(topics) >= max_topics:
            break
    return topics


def _format_gap(seconds: int, locale: str) -> str:
    """Format a time gap in seconds into a human-readable string."""
    if seconds < 3600:  # noqa: PLR2004
        minutes = max(1, seconds // 60)
        return f"{minutes} мин." if locale == "ru" else f"{minutes} min"
    hours = seconds // 3600
    if hours < 24:  # noqa: PLR2004
        return f"{hours} ч." if locale == "ru" else f"{hours}h"
    days = hours // 24
    return f"{days} дн." if locale == "ru" else f"{days}d"


async def build_continuity_hint(
    redis: Redis,
    user_id: str,
    messages: list[ChatMessage],
    locale: str | None = None,
) -> tuple[str, int]:
    """Build a continuity instruction if the user is returning after a break.

    Updates ``last_active:{user_id}`` in Redis.

    Returns:
        A tuple of (instruction_string, gap_seconds).  The instruction is
        empty when no hint is needed.  ``gap_seconds`` is the number of
        seconds since the user's last activity (0 when unknown), provided
        so that :func:`build_suggestion_hint` can reuse the value without
        re-reading the (now-updated) Redis key.
    """
    key = f"{_LAST_ACTIVE_PREFIX}:{user_id}"
    now = datetime.now(tz=UTC)
    now_ts = str(int(now.timestamp()))

    last_ts_raw = await redis.getset(key, now_ts)
    await redis.expire(key, 86400 * 7)  # 7-day expiry

    if last_ts_raw is None:
        return "", 0

    try:
        last_ts = int(last_ts_raw)
    except (ValueError, TypeError):
        return "", 0

    gap = int(now.timestamp()) - last_ts
    if gap < _CONTINUITY_GAP_SECONDS:
        return "", gap

    topics = _extract_recent_topics(messages)
    if not topics:
        return "", gap

    resolved = normalize_locale(locale)
    topics_str = "; ".join(topics)
    gap_str = _format_gap(gap, resolved)
    return tr(
        "prompt.continuity_instruction", resolved,
        topics=topics_str, gap=gap_str,
    ), gap


async def build_suggestion_hint(
    redis: Redis,
    user_id: str,
    system_prompt: str,
    locale: str | None = None,
    *,
    last_activity_gap: int = 0,
) -> str:
    """Build a proactive suggestion instruction if cooldown has expired.

    Checks the long-term profile for interests and the suggestion cooldown key.

    Args:
        last_activity_gap: Seconds since the user's last activity, as
            computed by :func:`build_continuity_hint`.  Avoids re-reading
            the ``last_active`` key which has already been updated to *now*.

    Returns an instruction string or empty string.
    """
    # Check cooldown
    cooldown_key = f"{_SUGGESTION_COOLDOWN_PREFIX}:{user_id}"
    exists = await redis.exists(cooldown_key)
    if exists:
        return ""

    # Use pre-computed gap from build_continuity_hint
    if last_activity_gap < _SUGGESTION_GAP_SECONDS:
        return ""

    # Extract interests from long-term profile
    ltp = extract_section(system_prompt, "Long-term Profile")
    if not ltp.strip():
        return ""

    # Look for interest-related lines
    interests: list[str] = []
    for line in ltp.splitlines():
        stripped = line.strip().lower()
        if any(keyword in stripped for keyword in ("interest", "like", "love", "enjoy", "hobby",
                                                    "интерес", "нрав", "люб", "хобби")):
            interests.append(line.strip())

    if not interests:
        return ""

    # Set cooldown
    await redis.set(cooldown_key, "1", ex=_SUGGESTION_COOLDOWN_TTL)

    resolved = normalize_locale(locale)
    interests_str = "; ".join(interests[:3])
    return tr("prompt.suggestion_instruction", resolved, interests=interests_str)


def _extract_interest_from_profile(profile_text: str) -> str:
    """Extract the user's interest from their long-term profile text."""
    for line in profile_text.splitlines():
        stripped = line.strip().lower()
        if "интересуется:" in stripped or "interested in:" in stripped:
            return line.strip().split(":", 1)[-1].strip()
    return ""


async def load_user_context(
    session: AsyncSession,
    snapshot_store: SnapshotStore,
    user_id: UUID,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    encryptor: FieldEncryptor | None = None,
    locale: str | None = None,
    redis: Redis | None = None,
    context_message_limit: int = 50,
) -> UserContext:
    """Build a :class:`~companion_bot_core.inference.schemas.UserContext` for *user_id*.

    Steps:
    1. Load the active prompt snapshot from the store; fall back to a minimal
       default if no snapshot exists yet.
    2. Load the recent non-expired message history from the database.
    3. Optionally inject continuity hints and proactive suggestions.
    4. Assemble and return :class:`~companion_bot_core.inference.schemas.UserContext`.

    Args:
        session:        Active database session (for message history).
        snapshot_store: Snapshot store (for active prompt snapshot).
        user_id:        Internal (UUID) user identifier.
        max_tokens:     Maximum completion tokens passed to the model.
        encryptor:      Optional field encryptor.
        locale:         User locale.
        redis:          Optional Redis client for continuity/suggestion hints.

    Returns:
        A populated :class:`~companion_bot_core.inference.schemas.UserContext`.
    """
    user_id_str = str(user_id)

    # Try prompt cache before hitting the snapshot store / DB.
    cached: dict[str, str] | None = None
    if redis is not None:
        try:
            cached = await get_cached_prompt(redis, user_id_str)
        except Exception:  # noqa: BLE001
            log.warning("prompt_cache_read_failed", user_id=user_id_str)

    if cached is not None:
        system_prompt = cached["system_prompt"]
    else:
        snapshot = await snapshot_store.get_active(user_id)

        if snapshot is not None:
            system_prompt = snapshot.system_prompt
        else:
            components = PromptComponents(base_system_template=DEFAULT_SYSTEM_TEMPLATE)
            system_prompt = build_system_prompt(components)
            # Persist an initial snapshot so the refinement worker has a base to
            # build on.  Without this, the worker always skips new users.
            version = await snapshot_store.next_version(user_id)
            initial = SnapshotRecord(
                user_id=user_id,
                version=version,
                system_prompt=system_prompt,
                source="initial",
            )
            await snapshot_store.save(initial, session=session)
            await snapshot_store.set_active(user_id, initial.id, session=session)

        # Populate cache on miss.
        if redis is not None:
            try:
                await cache_prompt(redis, user_id_str, {"system_prompt": system_prompt})
            except Exception:  # noqa: BLE001
                log.warning("prompt_cache_write_failed", user_id=user_id_str)

    if locale is not None:
        system_prompt = (
            f"{system_prompt}\n\n[Language]\n"
            f"{tr('prompt.language_instruction', normalize_locale(locale))}"
        )

    history = await load_recent_messages(
        session, user_id, limit=context_message_limit, encryptor=encryptor,
    )

    # Inject first-contact hint for new users (fewer than 3 messages = just finished onboarding)
    if len(history) < _FIRST_CONTACT_THRESHOLD:
        ltp = extract_section(system_prompt, "Long-term Profile")
        interest = _extract_interest_from_profile(ltp)
        resolved_locale = normalize_locale(locale)
        first_contact = tr(
            "prompt.first_contact_hint", resolved_locale,
            interest=interest,
        )
        system_prompt = f"{system_prompt}\n\n[FirstContact]\n{first_contact}"

    # Inject continuity hints, warm return, and proactive suggestions
    activity_gap = 0
    if redis is not None:
        try:
            continuity, activity_gap = await build_continuity_hint(
                redis, user_id_str, history, locale,
            )
        except Exception:  # noqa: BLE001
            continuity = ""
            log.warning("continuity_hint_failed", user_id=user_id_str)

        # Warm return: inject a stronger welcome-back hint for 48h+ gaps.
        # When warm return fires, suppress the weaker continuity hint to
        # avoid giving the model two conflicting return-acknowledgment
        # instructions.
        try:
            warm_hint = build_warm_return_hint(activity_gap, locale)
            if warm_hint:
                system_prompt = f"{system_prompt}\n\n[WarmReturn]\n{warm_hint}"
            elif continuity:
                system_prompt = f"{system_prompt}\n\n[Continuity]\n{continuity}"
        except Exception:  # noqa: BLE001
            if continuity:
                system_prompt = f"{system_prompt}\n\n[Continuity]\n{continuity}"
            log.warning("warm_return_hint_failed", user_id=user_id_str)

        try:
            suggestion = await build_suggestion_hint(
                redis, user_id_str, system_prompt, locale,
                last_activity_gap=activity_gap,
            )
            if suggestion:
                system_prompt = f"{system_prompt}\n\n[Suggestion]\n{suggestion}"
        except Exception:  # noqa: BLE001
            log.warning("suggestion_hint_failed", user_id=user_id_str)

    return UserContext(
        user_id=str(user_id),
        system_prompt=system_prompt,
        conversation_history=history,
        max_tokens=max_tokens,
    )
