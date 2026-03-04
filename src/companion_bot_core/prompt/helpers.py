"""Shared helpers for profile-to-snapshot operations.

Used by both ``bot.handlers`` (slash commands) and
``orchestrator.orchestrator`` (in-chat behavior detection) to avoid
duplicating the profile fetch, persona segment assembly, and snapshot
rebuild logic.
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from companion_bot_core.db.models import UserProfile
from companion_bot_core.prompt.merge_builder import (
    build_system_prompt,
    extract_base_template,
    extract_section,
)
from companion_bot_core.prompt.schemas import (
    DEFAULT_SYSTEM_TEMPLATE,
    PromptComponents,
    SnapshotRecord,
    SnapshotSource,
)

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from companion_bot_core.prompt.snapshot_store import SnapshotStore


async def acquire_profile_advisory_lock(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> None:
    """Acquire a PostgreSQL advisory transaction lock for the user's profile.

    Uses ``pg_advisory_xact_lock`` which blocks until the lock is available
    and is automatically released when the enclosing transaction commits or
    rolls back.  This provides a hard serialization guarantee — independent of
    any TTL — for the profile read-modify-write cycle.

    The lock key is a 64-bit signed integer derived from the full 128-bit UUID
    via BLAKE2b (8-byte digest).  A hash avoids the XOR symmetry flaw where two
    users whose UUID halves are byte-swapped would share the same key.  The
    Redis TTL lock in handlers still provides a fast-path rejection that avoids
    a DB round-trip when a concurrent request is clearly in-flight; this
    advisory lock covers the remaining edge case where the TTL expires before
    the transaction commits.
    """
    digest = hashlib.blake2b(user_id.bytes, digest_size=8).digest()
    lock_key = struct.unpack(">q", digest)[0]
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
        parts.append(f"Имя пользователя: {persona_name}")
    if tone:
        parts.append(f"Tone: {tone}")
    return "\n".join(parts)


def extract_memory_sections(
    snapshot: SnapshotRecord | None,
) -> dict[str, str]:
    """Extract human-readable memory sections from an active snapshot.

    Returns a dict with keys: ``persona``, ``tone``, ``skills``,
    ``long_term_profile``.  Values are empty strings when absent.
    """
    if snapshot is None:
        return {
            "persona": "",
            "tone": "",
            "skills": "",
            "long_term_profile": "",
        }

    persona_raw = extract_section(snapshot.system_prompt, "Persona")
    persona = ""
    tone = ""
    for line in persona_raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("Имя пользователя:"):
            persona = stripped[len("Имя пользователя:"):].strip()
        elif stripped.startswith("Name:"):
            # Legacy format — backwards compat
            persona = stripped[len("Name:"):].strip()
        elif stripped.startswith("Tone:"):
            tone = stripped[len("Tone:"):].strip()

    skills = (
        ", ".join(sorted(snapshot.skill_prompts_json.keys()))
        if snapshot.skill_prompts_json
        else ""
    )
    long_term_profile = extract_section(snapshot.system_prompt, "Long-term Profile")

    return {
        "persona": persona,
        "tone": tone,
        "skills": skills,
        "long_term_profile": long_term_profile,
    }


def _sanitize_fact(fact: str) -> str:
    """Strip section separators and header-like patterns from a user fact.

    Prevents section header injection (``[SectionName]``) and section separator
    injection (``---``) that could corrupt snapshot parsing.
    """
    lines: list[str] = []
    for line in fact.splitlines():
        stripped = line.strip()
        # Drop lines that look like section headers or separators
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        if stripped == "---":
            continue
        lines.append(line)
    return "\n".join(lines).strip()


async def add_fact_to_profile(
    snapshot_store: SnapshotStore,
    user_id: uuid.UUID,
    fact: str,
    *,
    session: Any = None,
) -> None:
    """Append a fact line to the long-term profile section and save a new snapshot."""
    fact = _sanitize_fact(fact)
    if not fact:
        return

    current = await snapshot_store.get_active(user_id)

    if current is not None:
        base_template = extract_base_template(current.system_prompt)
        skill_packs: dict[str, str] = {
            k: str(v) for k, v in current.skill_prompts_json.items()
        }
        long_term_profile = extract_section(current.system_prompt, "Long-term Profile")
        persona_segment = extract_section(current.system_prompt, "Persona")
        raw_skills: dict[str, object] = dict(current.skill_prompts_json)
    else:
        base_template = DEFAULT_SYSTEM_TEMPLATE
        skill_packs = {}
        long_term_profile = ""
        persona_segment = ""
        raw_skills = {}

    # Append the new fact with [manual] marker so the refinement worker
    # preserves user-supplied facts and avoids overwriting them.
    tagged_fact = f"[manual] {fact.strip()}"
    if long_term_profile.strip():
        long_term_profile = f"{long_term_profile.strip()}\n{tagged_fact}"
    else:
        long_term_profile = tagged_fact

    components = PromptComponents(
        base_system_template=base_template,
        persona_segment=persona_segment,
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
        source="user_command",
    )
    await snapshot_store.save(record, session=session)
    await snapshot_store.set_active(user_id, record.id, session=session)


async def remove_fact_from_profile(
    snapshot_store: SnapshotStore,
    user_id: uuid.UUID,
    query: str,
    *,
    session: Any = None,
) -> str | None:
    """Remove a matching fact line from the long-term profile.

    Returns the removed line text, or ``None`` if no match was found.
    Uses case-insensitive substring matching.
    """
    current = await snapshot_store.get_active(user_id)
    if current is None:
        return None

    base_template = extract_base_template(current.system_prompt)
    skill_packs: dict[str, str] = {
        k: str(v) for k, v in current.skill_prompts_json.items()
    }
    long_term_profile = extract_section(current.system_prompt, "Long-term Profile")
    persona_segment = extract_section(current.system_prompt, "Persona")
    raw_skills: dict[str, object] = dict(current.skill_prompts_json)

    if not long_term_profile.strip():
        return None

    lines = long_term_profile.strip().splitlines()
    query_lower = query.strip().lower()
    query_tokens = set(query_lower.split())
    removed_line: str | None = None
    remaining: list[str] = []

    for line in lines:
        line_lower = line.strip().lower()
        # Strip [manual] tag for matching purposes
        match_text = line_lower.removeprefix("[manual] ")
        # Match by substring or by token overlap (all query tokens appear)
        is_match = (
            query_lower in match_text
            or (query_tokens and query_tokens <= set(match_text.split()))
        )
        if removed_line is None and is_match:
            removed_line = line.strip()
            # Strip [manual] tag from the returned value for display
            if removed_line.startswith("[manual] "):
                removed_line = removed_line[len("[manual] "):]
        else:
            remaining.append(line)

    if removed_line is None:
        return None

    new_profile = "\n".join(remaining)

    components = PromptComponents(
        base_system_template=base_template,
        persona_segment=persona_segment,
        skill_packs=skill_packs,
        long_term_profile=new_profile,
    )
    system_prompt = build_system_prompt(components)

    version = await snapshot_store.next_version(user_id)
    record = SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt=system_prompt,
        skill_prompts_json=raw_skills,
        source="user_command",
    )
    await snapshot_store.save(record, session=session)
    await snapshot_store.set_active(user_id, record.id, session=session)
    return removed_line


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
