"""Unit tests for tdbot.prompt.rollback (rollback_to_previous, rollback_to_version)."""

from __future__ import annotations

import uuid

import pytest

from tdbot.prompt.rollback import RollbackError, rollback_to_previous, rollback_to_version
from tdbot.prompt.schemas import SnapshotRecord
from tdbot.prompt.snapshot_store import InMemorySnapshotStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seeded_store(
    user_id: uuid.UUID,
    prompts: list[str],
    *,
    set_active_version: int = 1,
) -> InMemorySnapshotStore:
    """Create a store with ``len(prompts)`` snapshots and set the active pointer."""
    store = InMemorySnapshotStore()
    snaps: list[SnapshotRecord] = []
    for v, prompt in enumerate(prompts, start=1):
        snap = SnapshotRecord(
            user_id=user_id,
            version=v,
            system_prompt=prompt,
            source="initial" if v == 1 else "user_command",
        )
        await store.save(snap)
        snaps.append(snap)

    target = next(s for s in snaps if s.version == set_active_version)
    await store.set_active(user_id, target.id)
    return store


# ---------------------------------------------------------------------------
# rollback_to_previous — basic
# ---------------------------------------------------------------------------


async def test_rollback_to_previous_creates_new_snapshot() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(user_id, ["v1-prompt", "v2-prompt"], set_active_version=2)

    result = await rollback_to_previous(store, user_id)

    assert result.source == "rollback"
    assert result.version == 3  # next after existing v1, v2


async def test_rollback_to_previous_copies_target_prompt() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(user_id, ["original", "updated"], set_active_version=2)

    result = await rollback_to_previous(store, user_id)
    assert result.system_prompt == "original"


async def test_rollback_to_previous_sets_active_pointer() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(user_id, ["v1", "v2"], set_active_version=2)

    rollback_snap = await rollback_to_previous(store, user_id)
    active = await store.get_active(user_id)
    assert active is not None
    assert active.id == rollback_snap.id


async def test_rollback_to_previous_preserves_skill_prompts() -> None:
    user_id = uuid.uuid4()
    store = InMemorySnapshotStore()
    snap1 = SnapshotRecord(
        user_id=user_id,
        version=1,
        system_prompt="p1",
        skill_prompts_json={"math": "solve equations"},
        source="initial",
    )
    snap2 = SnapshotRecord(
        user_id=user_id,
        version=2,
        system_prompt="p2",
        skill_prompts_json={},
        source="user_command",
    )
    await store.save(snap1)
    await store.save(snap2)
    await store.set_active(user_id, snap2.id)

    result = await rollback_to_previous(store, user_id)
    assert result.skill_prompts_json == {"math": "solve equations"}


async def test_rollback_to_previous_skill_prompts_are_copy() -> None:
    """Modifying the returned skill_prompts_json must not affect the target snapshot."""
    user_id = uuid.uuid4()
    store = InMemorySnapshotStore()
    snap1 = SnapshotRecord(
        user_id=user_id,
        version=1,
        system_prompt="p1",
        skill_prompts_json={"k": "v"},
        source="initial",
    )
    snap2 = SnapshotRecord(
        user_id=user_id, version=2, system_prompt="p2", source="user_command"
    )
    await store.save(snap1)
    await store.save(snap2)
    await store.set_active(user_id, snap2.id)

    result = await rollback_to_previous(store, user_id)
    result.skill_prompts_json["new_key"] = "tampered"

    # Original snap1 must be unaffected
    original = await store.get(snap1.id)
    assert original is not None
    assert "new_key" not in original.skill_prompts_json


# ---------------------------------------------------------------------------
# rollback_to_previous — error cases
# ---------------------------------------------------------------------------


async def test_rollback_to_previous_no_active_snapshot_raises() -> None:
    store = InMemorySnapshotStore()
    with pytest.raises(RollbackError, match="No active snapshot"):
        await rollback_to_previous(store, uuid.uuid4())


async def test_rollback_to_previous_only_one_snapshot_raises() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(user_id, ["only-one"], set_active_version=1)

    with pytest.raises(RollbackError, match="No previous snapshot found"):
        await rollback_to_previous(store, user_id)


async def test_rollback_to_previous_active_is_earliest_version_raises() -> None:
    user_id = uuid.uuid4()
    store = InMemorySnapshotStore()
    snap = SnapshotRecord(user_id=user_id, version=1, system_prompt="p", source="initial")
    await store.save(snap)
    await store.set_active(user_id, snap.id)
    # Manually add a newer snapshot but keep v1 active
    snap2 = SnapshotRecord(user_id=user_id, version=2, system_prompt="p2", source="refinement")
    await store.save(snap2)
    # Active is still v1 — no earlier version exists
    with pytest.raises(RollbackError, match="No previous snapshot found"):
        await rollback_to_previous(store, user_id)


# ---------------------------------------------------------------------------
# rollback_to_version — basic
# ---------------------------------------------------------------------------


async def test_rollback_to_version_restores_target() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(
        user_id, ["v1-text", "v2-text", "v3-text"], set_active_version=3
    )

    result = await rollback_to_version(store, user_id, target_version=1)
    assert result.system_prompt == "v1-text"
    assert result.source == "rollback"
    assert result.version == 4


async def test_rollback_to_version_updates_active_pointer() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(user_id, ["a", "b", "c"], set_active_version=3)

    rollback_snap = await rollback_to_version(store, user_id, target_version=2)
    active = await store.get_active(user_id)
    assert active is not None
    assert active.id == rollback_snap.id


async def test_rollback_to_version_nonexistent_raises() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(user_id, ["v1"], set_active_version=1)

    with pytest.raises(RollbackError, match="not found"):
        await rollback_to_version(store, user_id, target_version=99)


# ---------------------------------------------------------------------------
# History integrity after rollback
# ---------------------------------------------------------------------------


async def test_rollback_snapshot_persisted_in_store() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(user_id, ["a", "b"], set_active_version=2)

    rollback_snap = await rollback_to_previous(store, user_id)
    fetched = await store.get(rollback_snap.id)
    assert fetched is not None
    assert fetched.source == "rollback"


async def test_double_rollback_creates_two_new_snapshots() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(user_id, ["v1", "v2", "v3"], set_active_version=3)

    rb1 = await rollback_to_previous(store, user_id)
    rb2 = await rollback_to_previous(store, user_id)

    assert rb1.id != rb2.id
    assert rb2.version == rb1.version + 1


async def test_rollback_user_id_matches() -> None:
    user_id = uuid.uuid4()
    store = await _seeded_store(user_id, ["old", "new"], set_active_version=2)
    result = await rollback_to_previous(store, user_id)
    assert result.user_id == user_id
