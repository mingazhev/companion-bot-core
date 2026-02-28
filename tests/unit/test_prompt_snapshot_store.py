"""Unit tests for companion_bot_core.prompt.snapshot_store.InMemorySnapshotStore."""

from __future__ import annotations

import uuid

import pytest

from companion_bot_core.prompt.schemas import SnapshotRecord
from companion_bot_core.prompt.snapshot_store import InMemorySnapshotStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snap(user_id: uuid.UUID, version: int, prompt: str = "p") -> SnapshotRecord:
    return SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt=prompt,
        source="initial",
    )


# ---------------------------------------------------------------------------
# save / get
# ---------------------------------------------------------------------------


async def test_save_and_get_round_trip() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    snap = _make_snap(user_id, 1)
    await store.save(snap)
    fetched = await store.get(snap.id)
    assert fetched is not None
    assert fetched.id == snap.id
    assert fetched.version == 1


async def test_get_unknown_id_returns_none() -> None:
    store = InMemorySnapshotStore()
    result = await store.get(uuid.uuid4())
    assert result is None


async def test_save_twice_same_record_is_idempotent() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    snap = _make_snap(user_id, 1)
    await store.save(snap)
    await store.save(snap)  # second save with same UUID
    snaps = await store.list_for_user(user_id)
    assert len(snaps) == 1


# ---------------------------------------------------------------------------
# get_active / set_active
# ---------------------------------------------------------------------------


async def test_get_active_returns_none_for_new_user() -> None:
    store = InMemorySnapshotStore()
    result = await store.get_active(uuid.uuid4())
    assert result is None


async def test_set_active_then_get_active() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    snap = _make_snap(user_id, 1)
    await store.save(snap)
    await store.set_active(user_id, snap.id)
    active = await store.get_active(user_id)
    assert active is not None
    assert active.id == snap.id


async def test_set_active_replaces_previous_pointer() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    snap1 = _make_snap(user_id, 1)
    snap2 = _make_snap(user_id, 2)
    await store.save(snap1)
    await store.save(snap2)
    await store.set_active(user_id, snap1.id)
    await store.set_active(user_id, snap2.id)
    active = await store.get_active(user_id)
    assert active is not None
    assert active.id == snap2.id


async def test_set_active_raises_for_unknown_snapshot() -> None:
    store = InMemorySnapshotStore()
    with pytest.raises(KeyError):
        await store.set_active(uuid.uuid4(), uuid.uuid4())


# ---------------------------------------------------------------------------
# list_for_user
# ---------------------------------------------------------------------------


async def test_list_for_user_empty() -> None:
    store = InMemorySnapshotStore()
    result = await store.list_for_user(uuid.uuid4())
    assert result == []


async def test_list_for_user_newest_first() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    for v in (1, 2, 3):
        snap = _make_snap(user_id, v)
        await store.save(snap)

    result = await store.list_for_user(user_id)
    versions = [s.version for s in result]
    assert versions == [3, 2, 1]


async def test_list_for_user_respects_limit() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    for v in range(1, 6):
        await store.save(_make_snap(user_id, v))

    result = await store.list_for_user(user_id, limit=2)
    assert len(result) == 2
    # Should return the two newest
    assert result[0].version == 5
    assert result[1].version == 4


async def test_list_for_user_isolated_across_users() -> None:
    store = InMemorySnapshotStore()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    await store.save(_make_snap(user_a, 1))
    await store.save(_make_snap(user_b, 1))

    result_a = await store.list_for_user(user_a)
    result_b = await store.list_for_user(user_b)
    assert len(result_a) == 1
    assert len(result_b) == 1
    assert result_a[0].user_id == user_a
    assert result_b[0].user_id == user_b


# ---------------------------------------------------------------------------
# next_version
# ---------------------------------------------------------------------------


async def test_next_version_returns_one_for_new_user() -> None:
    store = InMemorySnapshotStore()
    assert await store.next_version(uuid.uuid4()) == 1


async def test_next_version_increments() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    await store.save(_make_snap(user_id, 1))
    assert await store.next_version(user_id) == 2


async def test_next_version_after_multiple_snapshots() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    for v in (1, 2, 5):  # non-contiguous versions
        await store.save(_make_snap(user_id, v))
    assert await store.next_version(user_id) == 6


# ---------------------------------------------------------------------------
# delete_for_user
# ---------------------------------------------------------------------------


async def test_delete_for_user_clears_active_pointer() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    snap = _make_snap(user_id, 1)
    await store.save(snap)
    await store.set_active(user_id, snap.id)
    assert await store.get_active(user_id) is not None

    await store.delete_for_user(user_id)
    assert await store.get_active(user_id) is None


async def test_delete_for_user_removes_all_snapshots() -> None:
    store = InMemorySnapshotStore()
    user_id = uuid.uuid4()
    snaps = [_make_snap(user_id, v) for v in (1, 2, 3)]
    for s in snaps:
        await store.save(s)

    await store.delete_for_user(user_id)

    assert await store.list_for_user(user_id) == []
    for s in snaps:
        assert await store.get(s.id) is None


async def test_delete_for_user_does_not_affect_other_users() -> None:
    store = InMemorySnapshotStore()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    snap_a = _make_snap(user_a, 1)
    snap_b = _make_snap(user_b, 1)
    await store.save(snap_a)
    await store.save(snap_b)
    await store.set_active(user_a, snap_a.id)
    await store.set_active(user_b, snap_b.id)

    await store.delete_for_user(user_a)

    assert await store.get_active(user_a) is None
    assert await store.get_active(user_b) is not None
    assert await store.get(snap_b.id) is not None
