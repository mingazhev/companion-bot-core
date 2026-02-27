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

    async def get_active(self, user_id: UUID) -> SnapshotRecord | None:
        """Return the currently active snapshot for *user_id*.

        The active pointer is stored in Redis.  Returns ``None`` when no
        pointer exists or when the referenced snapshot cannot be found.
        """
        raw = await self._redis.get(f"prompt:active:{user_id}")
        if raw is None:
            return None
        try:
            snapshot_id = UUID(raw)
        except ValueError:
            log.warning("invalid_active_pointer", user_id=str(user_id), raw=raw)
            return None
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


async def flush_deferred_redis_writes(
    session: AsyncSession,
    redis: Redis,
) -> None:
    """Execute snapshot-pointer Redis writes deferred during *session*.

    Must be called **after** the DB transaction has committed.  If no
    deferred writes were recorded the call is a no-op.

    Writes are removed from ``session.info`` only after all Redis SETs
    succeed so that callers can retry on transient failures.
    """
    writes: list[tuple[str, str]] = session.info.get(_DEFERRED_REDIS_KEY, [])
    if not writes:
        return
    for key, value in writes:
        await redis.set(key, value)
    # All writes succeeded — safe to clear.
    session.info.pop(_DEFERRED_REDIS_KEY, None)
