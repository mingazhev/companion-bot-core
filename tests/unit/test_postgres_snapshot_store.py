"""Unit tests for PostgresSnapshotStore.

These tests use mocked DB sessions and Redis to verify the store's behaviour
without external dependencies.  Integration tests with real PostgreSQL and
Redis would live in tests/integration/.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tdbot.prompt.postgres_store import (
    PostgresSnapshotStore,
    extract_deferred_redis_writes,
)
from tdbot.prompt.schemas import SnapshotRecord

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_record(
    user_id: uuid.UUID | None = None,
    version: int = 1,
    source: str = "initial",
) -> SnapshotRecord:
    return SnapshotRecord(
        user_id=user_id or uuid.uuid4(),
        version=version,
        system_prompt="You are a test companion.",
        source=source,  # type: ignore[arg-type]
    )


def _make_orm_snapshot(record: SnapshotRecord) -> MagicMock:
    """Create a mock ORM PromptSnapshot from a SnapshotRecord."""
    orm = MagicMock()
    orm.id = record.id
    orm.user_id = record.user_id
    orm.version = record.version
    orm.system_prompt = record.system_prompt
    orm.skill_prompts_json = dict(record.skill_prompts_json)
    orm.source = record.source
    orm.created_at = record.created_at
    return orm


def _make_store() -> tuple[PostgresSnapshotStore, AsyncMock, AsyncMock]:
    """Create a PostgresSnapshotStore with mocked engine and Redis."""
    engine = AsyncMock()
    redis = AsyncMock()
    store = PostgresSnapshotStore(engine=engine, redis=redis)
    return store, engine, redis


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_save_persists_to_db() -> None:
    store, engine, _redis = _make_store()
    record = _make_record()

    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    with patch("tdbot.prompt.postgres_store.get_async_session") as mock_get_session:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        await store.save(record)

    mock_session.add.assert_called_once()
    added_orm = mock_session.add.call_args[0][0]
    assert added_orm.id == record.id
    assert added_orm.user_id == record.user_id
    assert added_orm.system_prompt == record.system_prompt


# --------------------------------------------------------------------------- #
# get
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_returns_record_when_found() -> None:
    store, engine, _redis = _make_store()
    record = _make_record()
    orm_row = _make_orm_snapshot(record)

    mock_session = AsyncMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = orm_row
    mock_session.execute = AsyncMock(return_value=exec_result)

    with patch("tdbot.prompt.postgres_store.get_async_session") as mock_get_session:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        result = await store.get(record.id)

    assert result is not None
    assert result.id == record.id
    assert result.system_prompt == record.system_prompt


@pytest.mark.asyncio
async def test_get_returns_none_when_not_found() -> None:
    store, engine, _redis = _make_store()

    mock_session = AsyncMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=exec_result)

    with patch("tdbot.prompt.postgres_store.get_async_session") as mock_get_session:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        result = await store.get(uuid.uuid4())

    assert result is None


# --------------------------------------------------------------------------- #
# get_active / set_active
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_active_returns_none_when_no_pointer() -> None:
    store, _engine, redis = _make_store()
    redis.get = AsyncMock(return_value=None)

    result = await store.get_active(uuid.uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_set_active_raises_key_error_for_missing_snapshot() -> None:
    store, engine, redis = _make_store()
    user_id = uuid.uuid4()
    snapshot_id = uuid.uuid4()

    # Mock get() to return None (snapshot not found)
    mock_session = AsyncMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=exec_result)

    with (
        patch("tdbot.prompt.postgres_store.get_async_session") as mock_get_session,
        pytest.raises(KeyError, match=str(snapshot_id)),
    ):
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        await store.set_active(user_id, snapshot_id)


@pytest.mark.asyncio
async def test_set_active_sets_redis_key() -> None:
    store, engine, redis = _make_store()
    record = _make_record()
    orm_row = _make_orm_snapshot(record)

    mock_session = AsyncMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = orm_row
    mock_session.execute = AsyncMock(return_value=exec_result)
    redis.set = AsyncMock()

    with patch("tdbot.prompt.postgres_store.get_async_session") as mock_get_session:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        await store.set_active(record.user_id, record.id)

    redis.set.assert_awaited_once_with(
        f"prompt:active:{record.user_id}", str(record.id)
    )


# --------------------------------------------------------------------------- #
# next_version
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_next_version_uses_redis_incr() -> None:
    store, _engine, redis = _make_store()
    user_id = uuid.uuid4()
    redis.incr = AsyncMock(return_value=3)

    version = await store.next_version(user_id)

    assert version == 3
    redis.incr.assert_awaited_once_with(f"prompt:version:{user_id}")


# --------------------------------------------------------------------------- #
# list_for_user
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_for_user_returns_records_newest_first() -> None:
    store, engine, _redis = _make_store()
    user_id = uuid.uuid4()
    r1 = _make_record(user_id=user_id, version=1)
    r2 = _make_record(user_id=user_id, version=2)
    orm1 = _make_orm_snapshot(r1)
    orm2 = _make_orm_snapshot(r2)

    mock_session = AsyncMock()
    exec_result = MagicMock()
    scalars_result = MagicMock()
    scalars_result.all.return_value = [orm2, orm1]  # newest first
    exec_result.scalars.return_value = scalars_result
    mock_session.execute = AsyncMock(return_value=exec_result)

    with patch("tdbot.prompt.postgres_store.get_async_session") as mock_get_session:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        records = await store.list_for_user(user_id)

    assert len(records) == 2
    assert records[0].version == 2
    assert records[1].version == 1


# --------------------------------------------------------------------------- #
# delete_for_user
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_delete_for_user_cleans_redis_keys() -> None:
    """delete_for_user only removes Redis keys; DB rows are cascade-deleted."""
    store, _engine, redis = _make_store()
    user_id = uuid.uuid4()
    redis.delete = AsyncMock()

    await store.delete_for_user(user_id)

    # Only Redis keys are cleaned (DB rows rely on ON DELETE CASCADE).
    redis.delete.assert_awaited_once_with(
        f"prompt:active:{user_id}",
        f"prompt:version:{user_id}",
    )


# --------------------------------------------------------------------------- #
# extract_deferred_redis_writes
# --------------------------------------------------------------------------- #


_DEFERRED_KEY = "_snapshot_deferred_redis_writes"


def test_extract_deferred_redis_writes_returns_and_clears() -> None:
    """extract_deferred_redis_writes pops the deferred list from session.info."""
    session = MagicMock()
    session.info = {_DEFERRED_KEY: [("prompt:active:x", "snap-1")]}
    result = extract_deferred_redis_writes(session)
    assert result == [("prompt:active:x", "snap-1")]
    assert _DEFERRED_KEY not in session.info


def test_extract_deferred_redis_writes_returns_empty_when_no_key() -> None:
    """extract_deferred_redis_writes returns [] when no deferred writes are queued."""
    session = MagicMock()
    session.info = {}
    result = extract_deferred_redis_writes(session)
    assert result == []


def test_extract_deferred_redis_writes_is_idempotent_on_second_call() -> None:
    """Calling extract_deferred_redis_writes twice on the same session returns []
    on the second call — confirming pop semantics, not get."""
    session = MagicMock()
    session.info = {_DEFERRED_KEY: [("prompt:active:x", "snap-1")]}
    first = extract_deferred_redis_writes(session)
    second = extract_deferred_redis_writes(session)
    assert first == [("prompt:active:x", "snap-1")]
    assert second == []
