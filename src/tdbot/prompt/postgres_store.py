"""PostgreSQL + Redis backed snapshot store for production use.

Snapshots are persisted to the ``prompt_snapshots`` table.  The **active
pointer** (which snapshot is current for a user) is stored in Redis as
``prompt:active:<user_id>`` for sub-millisecond reads.  The **version
counter** uses ``prompt:version:<user_id>`` with Redis ``INCR`` for strict
monotonicity under concurrent writers.

This module satisfies the :class:`~tdbot.prompt.snapshot_store.SnapshotStore`
protocol and is intended for production deployments.  Use
:class:`~tdbot.prompt.snapshot_store.InMemorySnapshotStore` for unit tests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from tdbot.db.engine import get_async_session
from tdbot.db.models import PromptSnapshot
from tdbot.logging_config import get_logger
from tdbot.prompt.schemas import SnapshotRecord

log = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

# Key used in ``session.info`` to accumulate Redis writes that must be
# executed only after the DB transaction commits.
_DEFERRED_REDIS_KEY = "_snapshot_deferred_redis_writes"


def _to_record(row: PromptSnapshot) -> SnapshotRecord:
    """Convert an ORM ``PromptSnapshot`` row to a ``SnapshotRecord``."""
    return SnapshotRecord(
        id=row.id,
        user_id=row.user_id,
        version=row.version,
        system_prompt=row.system_prompt,
        skill_prompts_json=dict(row.skill_prompts_json),
        source=row.source,  # type: ignore[arg-type]
        created_at=row.created_at,
    )


class PostgresSnapshotStore:
    """Production snapshot store backed by PostgreSQL and Redis.

    Satisfies the ``SnapshotStore`` protocol structurally.
    """

    def __init__(self, engine: AsyncEngine, redis: Redis) -> None:
        self._engine = engine
        self._redis = redis

    @asynccontextmanager
    async def _use_session(
        self, session: AsyncSession | None,
    ) -> AsyncGenerator[AsyncSession, None]:
        """Yield *session* when provided, otherwise open a private one."""
        if session is not None:
            yield session
        else:
            async with get_async_session(self._engine) as own_session:
                yield own_session

    # -- persistence -------------------------------------------------------- #

    async def save(
        self,
        record: SnapshotRecord,
        session: AsyncSession | None = None,
    ) -> None:
        """Persist *record* to PostgreSQL.

        When *session* is provided the row is added to that session (committed
        by the caller).  Otherwise a private session is opened and committed
        immediately.
        """
        orm_row = PromptSnapshot(
            id=record.id,
            user_id=record.user_id,
            version=record.version,
            system_prompt=record.system_prompt,
            skill_prompts_json=dict(record.skill_prompts_json),
            source=record.source,
        )
        async with self._use_session(session) as s:
            s.add(orm_row)

    async def get(
        self,
        snapshot_id: UUID,
        session: AsyncSession | None = None,
    ) -> SnapshotRecord | None:
        """Fetch a snapshot by UUID from PostgreSQL."""
        async with self._use_session(session) as s:
            result = await s.execute(
                select(PromptSnapshot).where(PromptSnapshot.id == snapshot_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _to_record(row)

    # -- active pointer (Redis) --------------------------------------------- #

    async def _get_latest_from_db(self, user_id: UUID) -> SnapshotRecord | None:
        """Query the DB for the snapshot with the highest version for *user_id*.

        Used as a fallback when the Redis active pointer is missing — for
        example when the deferred pointer write failed after a DB commit, or
        on first access before any pointer has been written.  The highest
        version is the most recently saved snapshot, which is correct for all
        non-rollback cases.  A rollback re-writes the Redis pointer directly,
        so it would only be absent here if the rollback's deferred write also
        failed (an independently unlikely event).
        """
        async with get_async_session(self._engine) as session:
            result = await session.execute(
                select(PromptSnapshot)
                .where(PromptSnapshot.user_id == user_id)
                .order_by(PromptSnapshot.version.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _to_record(row)

    async def get_active(self, user_id: UUID) -> SnapshotRecord | None:
        """Return the currently active snapshot for *user_id*.

        The active pointer is stored in Redis for low-latency reads.  When the
        pointer is absent (e.g. never written, or deferred flush failed after
        commit), the method falls back to the DB and returns the snapshot with
        the highest version for the user.  Returns ``None`` only when no
        snapshot exists in either Redis or the DB.
        """
        raw = await self._redis.get(f"prompt:active:{user_id}")
        if raw is None:
            # Redis pointer missing — fall back to DB.
            return await self._get_latest_from_db(user_id)
        try:
            snapshot_id = UUID(raw)
        except ValueError:
            log.warning("invalid_active_pointer", user_id=str(user_id), raw=raw)
            return await self._get_latest_from_db(user_id)
        return await self.get(snapshot_id)

    async def set_active(
        self,
        user_id: UUID,
        snapshot_id: UUID,
        session: AsyncSession | None = None,
    ) -> None:
        """Atomically update the active pointer for *user_id* in Redis.

        When *session* is provided the Redis write is **deferred** — it is
        stored in ``session.info`` and must be flushed by the caller (via
        :func:`flush_deferred_redis_writes`) after the transaction commits.
        This prevents the Redis pointer from referencing an uncommitted or
        rolled-back snapshot row.

        When *session* is ``None`` the pointer is written immediately (the
        snapshot is assumed to already be committed).

        Raises :class:`KeyError` when *snapshot_id* is not found in the store.
        """
        record = await self.get(snapshot_id, session=session)
        if record is None:
            raise KeyError(f"Snapshot {snapshot_id} not found in store")

        if session is not None:
            pending = session.info.setdefault(_DEFERRED_REDIS_KEY, [])
            pending.append((f"prompt:active:{user_id}", str(snapshot_id)))
        else:
            await self._redis.set(f"prompt:active:{user_id}", str(snapshot_id))

    # -- listing / versioning ----------------------------------------------- #

    async def list_for_user(
        self, user_id: UUID, *, limit: int = 50
    ) -> list[SnapshotRecord]:
        """Return snapshots for *user_id* ordered newest-first (version desc)."""
        async with get_async_session(self._engine) as session:
            result = await session.execute(
                select(PromptSnapshot)
                .where(PromptSnapshot.user_id == user_id)
                .order_by(PromptSnapshot.version.desc())
                .limit(limit)
            )
            return [_to_record(row) for row in result.scalars().all()]

    async def next_version(self, user_id: UUID) -> int:
        """Return the next monotonic version number via Redis INCR."""
        result = await self._redis.incr(f"prompt:version:{user_id}")
        return int(result)

    # -- cleanup ------------------------------------------------------------ #

    async def delete_for_user(self, user_id: UUID) -> None:
        """Remove Redis keys for *user_id*.

        DB rows are cascade-deleted when the ``User`` row is removed by
        :func:`~tdbot.privacy.delete_user.hard_delete_user`, so this method
        only cleans up the Redis active-pointer and version-counter keys.
        Opening a separate DB session here would deadlock when the caller
        already holds a ``FOR UPDATE`` lock on the user row within an
        uncommitted transaction.
        """
        await self._redis.delete(
            f"prompt:active:{user_id}",
            f"prompt:version:{user_id}",
        )


def extract_deferred_redis_writes(session: AsyncSession) -> list[tuple[str, str]]:
    """Pop deferred Redis writes from *session*.info and return them.

    Must be called **inside** the session context manager, before the
    session is closed.  The returned list should then be passed to
    :func:`flush_deferred_redis_writes` after the transaction commits.
    """
    result: list[tuple[str, str]] = session.info.pop(_DEFERRED_REDIS_KEY, [])
    return result


async def flush_deferred_redis_writes(
    writes: list[tuple[str, str]],
    redis: Redis,
) -> None:
    """Execute snapshot-pointer Redis writes previously extracted via
    :func:`extract_deferred_redis_writes`.

    Must be called **after** the DB transaction has committed.  If
    *writes* is empty the call is a no-op.  Redis ``SET`` is
    idempotent so callers can safely retry on transient failures.
    """
    for key, value in writes:
        await redis.set(key, value)


# --------------------------------------------------------------------------- #
# Profile write-lock deferral helpers
# --------------------------------------------------------------------------- #

# Lua script for ownership-safe Redis lock release.  Deletes the key only if
# its current value matches the caller's unique token, preventing a request
# from releasing a lock it no longer owns (e.g. after the TTL expired and a
# second request re-acquired the same key).  Runs atomically on the Redis server.
PROFILE_LOCK_UNLOCK_SCRIPT = (
    "if redis.call('GET', KEYS[1]) == ARGV[1] then "
    "return redis.call('DEL', KEYS[1]) "
    "else return 0 end"
)

# Key used in ``session.info`` to accumulate profile lock releases that must
# be executed only after the DB transaction commits and deferred Redis pointer
# writes are flushed.  This prevents a concurrent request from acquiring the
# lock and reading stale DB state or a stale Redis active pointer.
_DEFERRED_LOCK_RELEASES_KEY = "_profile_lock_releases"


def defer_lock_release(
    session: AsyncSession, lock_key: str, lock_token: str
) -> None:
    """Queue an ownership-safe lock release to execute after the DB transaction
    commits and deferred Redis pointer writes are flushed.

    Must be called **inside** the session context manager.  The lock info is
    stored in ``session.info`` and extracted by the caller via
    :func:`extract_deferred_lock_releases`.
    """
    pending: list[tuple[str, str]] = session.info.setdefault(
        _DEFERRED_LOCK_RELEASES_KEY, []
    )
    pending.append((lock_key, lock_token))


def extract_deferred_lock_releases(session: AsyncSession) -> list[tuple[str, str]]:
    """Pop deferred lock releases from *session*.info and return them.

    Must be called **inside** the session context manager, before the session
    is closed.  The returned list should then be passed to
    :func:`flush_deferred_lock_releases` after the transaction commits and
    after deferred Redis pointer writes are flushed.
    """
    return session.info.pop(_DEFERRED_LOCK_RELEASES_KEY, [])


async def flush_deferred_lock_releases(
    releases: list[tuple[str, str]],
    redis: Redis,  # type: ignore[type-arg]
) -> None:
    """Run ownership-safe lock releases previously extracted via
    :func:`extract_deferred_lock_releases`.

    Must be called **after** the DB transaction has committed and after
    deferred Redis pointer writes are flushed, so that any request that
    acquires the lock immediately afterwards sees fully committed state.
    If *releases* is empty the call is a no-op.
    """
    for lock_key, lock_token in releases:
        await redis.eval(PROFILE_LOCK_UNLOCK_SCRIPT, 1, lock_key, lock_token)  # type: ignore[no-untyped-call]
