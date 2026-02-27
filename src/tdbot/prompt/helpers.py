"""Shared helpers for profile-to-snapshot operations.

Used by both ``bot.handlers`` (slash commands) and
``orchestrator.orchestrator`` (in-chat behavior detection) to avoid
duplicating the profile fetch, persona segment assembly, and snapshot
rebuild logic.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from tdbot.db.models import UserProfile
from tdbot.prompt.merge_builder import build_system_prompt, extract_base_template, extract_section
from tdbot.prompt.schemas import (
    DEFAULT_SYSTEM_TEMPLATE,
    PromptComponents,
    SnapshotRecord,
    SnapshotSource,
)

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from tdbot.prompt.snapshot_store import SnapshotStore


async def acquire_profile_advisory_lock(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> None:
    """Acquire a PostgreSQL advisory transaction lock for the user's profile.

    Uses ``pg_advisory_xact_lock`` which blocks until the lock is available
    and is automatically released when the enclosing transaction commits or
    rolls back.  This provides a hard serialization guarantee — independent of
    any TTL — for the profile read-modify-write cycle.

    The lock key is derived by XOR-ing the two 8-byte halves of the user UUID
    and mapping the result to a signed int64 (big-endian).  Using the full 128
    bits avoids advisory-lock key collisions between users whose UUIDs share
    the same first 8 bytes.  The Redis TTL lock in handlers still provides a
    fast-path rejection that avoids a DB round-trip when a concurrent request
    is clearly in-flight; this advisory lock covers the remaining edge case
    where the TTL expires before the transaction commits.
    """
    b = user_id.bytes
    hi = struct.unpack(">q", b[:8])[0]
    lo = struct.unpack(">q", b[8:])[0]
    lock_key = hi ^ lo
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key}
    )


async def get_or_create_profile(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> UserProfile:
    """Fetch the user's profile row, creating one if it doesn't exist.

    Uses an upsert (INSERT … ON CONFLICT DO NOTHING) to avoid a race
    condition when two concurrent requests arrive for the same user,
    mirroring the pattern in :func:`bot.users.get_or_create_user`.
    """
    stmt = (
        insert(UserProfile)
        .values(user_id=user_id)
        .on_conflict_do_nothing(index_elements=["user_id"])
    )
    await session.execute(stmt)
    result = await session.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    return result.scalar_one()


def build_persona_segment(
    persona_name: str | None,
    tone: str | None,
) -> str:
    """Build the persona section content from profile fields."""
    parts: list[str] = []
    if persona_name:
        parts.append(f"Name: {persona_name}")
    if tone:
        parts.append(f"Tone: {tone}")
    return "\n".join(parts)


async def rebuild_and_save_snapshot(
    snapshot_store: SnapshotStore,
    user_id: uuid.UUID,
    persona_name: str | None,
    tone: str | None,
    *,
    source: SnapshotSource = "user_command",
    session: Any = None,
) -> None:
    """Build a new prompt snapshot reflecting updated profile and set it active.

    Args:
        snapshot_store: The snapshot store to save into.
        user_id:        Internal UUID of the user.
        persona_name:   Decrypted persona name (or ``None``).
        tone:           Decrypted tone (or ``None``).
        source:         Source label for the snapshot (``"user_command"`` or
                        ``"behavior_change"``).
        session:        Optional DB session.  When provided, the snapshot row
                        is added to this session so it commits atomically with
                        the caller's transaction.
    """
    current = await snapshot_store.get_active(user_id)

    raw_skills: dict[str, object] = {}
    if current is not None:
        base_template = extract_base_template(current.system_prompt)
        skill_packs: dict[str, str] = {
            k: str(v) for k, v in current.skill_prompts_json.items()
        }
        long_term_profile = extract_section(current.system_prompt, "Long-term Profile")
        raw_skills = dict(current.skill_prompts_json)
    else:
        base_template = DEFAULT_SYSTEM_TEMPLATE
        skill_packs = {}
        long_term_profile = ""

    components = PromptComponents(
        base_system_template=base_template,
        persona_segment=build_persona_segment(persona_name, tone),
        skill_packs=skill_packs,
        long_term_profile=long_term_profile,
    )
    system_prompt = build_system_prompt(components)

    version = await snapshot_store.next_version(user_id)
    record = SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt=system_prompt,
        skill_prompts_json=raw_skills,
        source=source,
    )
    await snapshot_store.save(record, session=session)
    await snapshot_store.set_active(user_id, record.id, session=session)
