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

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import delete, select

from tdbot.db.engine import get_async_session
from tdbot.db.models import PromptSnapshot
from tdbot.prompt.schemas import SnapshotRecord

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine


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

    # -- persistence -------------------------------------------------------- #

    async def save(self, record: SnapshotRecord) -> None:
        """Persist *record* to PostgreSQL."""
        async with get_async_session(self._engine) as session:
            orm_row = PromptSnapshot(
                id=record.id,
                user_id=record.user_id,
                version=record.version,
                system_prompt=record.system_prompt,
                skill_prompts_json=dict(record.skill_prompts_json),
                source=record.source,
            )
            session.add(orm_row)

    async def get(self, snapshot_id: UUID) -> SnapshotRecord | None:
        """Fetch a snapshot by UUID from PostgreSQL."""
        async with get_async_session(self._engine) as session:
            result = await session.execute(
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
        snapshot_id = UUID(raw.decode() if isinstance(raw, bytes) else raw)
        return await self.get(snapshot_id)

    async def set_active(self, user_id: UUID, snapshot_id: UUID) -> None:
        """Atomically update the active pointer for *user_id* in Redis.

        Raises :class:`KeyError` when *snapshot_id* is not found in the store.
        """
        record = await self.get(snapshot_id)
        if record is None:
            raise KeyError(f"Snapshot {snapshot_id} not found in store")
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
        """Remove all snapshots from PostgreSQL and Redis keys for *user_id*."""
        async with get_async_session(self._engine) as session:
            await session.execute(
                delete(PromptSnapshot).where(PromptSnapshot.user_id == user_id)
            )
        await self._redis.delete(
            f"prompt:active:{user_id}",
            f"prompt:version:{user_id}",
        )
