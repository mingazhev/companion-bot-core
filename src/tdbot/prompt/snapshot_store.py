"""Snapshot store protocol and in-memory reference implementation.

Production notes
----------------
- **Snapshots** are persisted to PostgreSQL via the ``prompt_snapshots`` ORM model.
- The **active pointer** is stored in Redis as the key
  ``prompt:active:<user_id>`` (value = snapshot UUID string) so that reads are
  sub-millisecond and updates are atomic (Redis ``SET`` is single-threaded on the
  server).  No locking is needed in Python code for the pointer itself.
- ``next_version`` in production uses a Redis counter
  ``prompt:version:<user_id>`` with ``INCR`` for strict monotonicity even under
  concurrent writers.

For unit tests ``InMemorySnapshotStore`` provides the same interface with no
external dependencies.  Because asyncio is cooperatively scheduled, plain dict
operations are effectively atomic for test purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import uuid

    from tdbot.prompt.schemas import SnapshotRecord


class SnapshotStore(Protocol):
    """Async snapshot store used by the prompt state manager."""

    async def save(self, record: SnapshotRecord) -> None:
        """Persist a new snapshot (must be immutable after save)."""
        ...

    async def get(self, snapshot_id: uuid.UUID) -> SnapshotRecord | None:
        """Fetch a snapshot by UUID; return None if not found."""
        ...

    async def get_active(self, user_id: uuid.UUID) -> SnapshotRecord | None:
        """Return the currently active snapshot for *user_id*, or None."""
        ...

    async def set_active(self, user_id: uuid.UUID, snapshot_id: uuid.UUID) -> None:
        """Atomically update the active pointer for *user_id*.

        In production: ``SET prompt:active:<user_id> <snapshot_id>`` in Redis.

        Raises:
            KeyError: When *snapshot_id* is not known to the store.
        """
        ...

    async def list_for_user(
        self, user_id: uuid.UUID, *, limit: int = 50
    ) -> list[SnapshotRecord]:
        """Return snapshots for *user_id* ordered **newest-first** (version desc)."""
        ...

    async def next_version(self, user_id: uuid.UUID) -> int:
        """Return the next monotonic version number for *user_id*.

        In production: Redis ``INCR prompt:version:<user_id>``.
        """
        ...

    async def delete_for_user(self, user_id: uuid.UUID) -> None:
        """Remove all snapshots and the active pointer for *user_id*."""
        ...


class InMemorySnapshotStore:
    """In-memory snapshot store for unit testing.

    Satisfies ``SnapshotStore`` structurally (duck-typing / Protocol).
    No external dependencies; all state is held in plain dicts.
    """

    def __init__(self) -> None:
        # snapshot_id -> record
        self._snapshots: dict[uuid.UUID, SnapshotRecord] = {}
        # user_id -> ordered list of snapshot UUIDs (insertion order = version order)
        self._user_index: dict[uuid.UUID, list[uuid.UUID]] = {}
        # user_id -> active snapshot UUID
        self._active: dict[uuid.UUID, uuid.UUID] = {}

    async def save(self, record: SnapshotRecord) -> None:
        """Persist *record*.  Silently overwrites if the same UUID is saved twice."""
        self._snapshots[record.id] = record
        bucket = self._user_index.setdefault(record.user_id, [])
        if record.id not in bucket:
            bucket.append(record.id)

    async def get(self, snapshot_id: uuid.UUID) -> SnapshotRecord | None:
        return self._snapshots.get(snapshot_id)

    async def get_active(self, user_id: uuid.UUID) -> SnapshotRecord | None:
        active_id = self._active.get(user_id)
        if active_id is None:
            return None
        return self._snapshots.get(active_id)

    async def set_active(self, user_id: uuid.UUID, snapshot_id: uuid.UUID) -> None:
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot {snapshot_id} not found in store")
        self._active[user_id] = snapshot_id

    async def list_for_user(
        self, user_id: uuid.UUID, *, limit: int = 50
    ) -> list[SnapshotRecord]:
        ids = self._user_index.get(user_id, [])
        # Newest first: reverse the insertion-order list (which tracks version order)
        records = [
            self._snapshots[sid] for sid in reversed(ids) if sid in self._snapshots
        ]
        return records[:limit]

    async def next_version(self, user_id: uuid.UUID) -> int:
        ids = self._user_index.get(user_id, [])
        if not ids:
            return 1
        versions = [
            self._snapshots[sid].version for sid in ids if sid in self._snapshots
        ]
        return max(versions, default=0) + 1

    async def delete_for_user(self, user_id: uuid.UUID) -> None:
        """Remove all snapshots and the active pointer for *user_id*."""
        ids = self._user_index.pop(user_id, [])
        for sid in ids:
            self._snapshots.pop(sid, None)
        self._active.pop(user_id, None)
