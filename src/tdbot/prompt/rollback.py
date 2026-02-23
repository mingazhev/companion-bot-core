"""Prompt snapshot rollback logic.

Rollback creates a *new* immutable snapshot that is a copy of the target
historical snapshot with ``source="rollback"``.  This keeps the full audit
trail intact — the history always shows the original snapshots plus an
explicit rollback entry.

Two public functions are provided:

    rollback_to_previous  — revert one step to the snapshot just before the
                            current active one (the common case).
    rollback_to_version   — revert to an explicit version number (used by the
                            orchestrator after a failed quality check).

Both functions validate the target, create the rollback snapshot, and
atomically update the active pointer via ``store.set_active``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tdbot.metrics import PROMPT_ROLLBACKS
from tdbot.prompt.schemas import SnapshotRecord

if TYPE_CHECKING:
    import uuid

    from tdbot.prompt.snapshot_store import SnapshotStore


class RollbackError(Exception):
    """Raised when a rollback cannot be performed."""


async def rollback_to_previous(
    store: SnapshotStore,
    user_id: uuid.UUID,
    *,
    reason: str = "user_command",
) -> SnapshotRecord:
    """Roll back the active snapshot one step to the previous version.

    Args:
        store:   Snapshot store holding the user's snapshot history.
        user_id: Target user.
        reason:  Reason label for the rollback metric
                 (``"user_command"``, ``"quality_check"``, or ``"manual"``).

    Returns:
        The newly created rollback ``SnapshotRecord`` (now active).

    Raises:
        RollbackError: When there is no active snapshot or no earlier version
                       exists to roll back to.
    """
    active = await store.get_active(user_id)
    if active is None:
        raise RollbackError("No active snapshot to roll back from")

    history = await store.list_for_user(user_id, limit=50)
    # history is newest-first; find the first entry with a strictly lower version
    target: SnapshotRecord | None = None
    for snap in history:
        if snap.version < active.version:
            target = snap
            break

    if target is None:
        raise RollbackError(
            f"No previous snapshot found for user {user_id}; "
            "cannot roll back further"
        )

    return await _apply_rollback(store, user_id, target, reason=reason)


async def rollback_to_version(
    store: SnapshotStore,
    user_id: uuid.UUID,
    *,
    target_version: int,
    reason: str = "manual",
) -> SnapshotRecord:
    """Roll back the active snapshot to a specific historical version.

    Args:
        store:          Snapshot store holding the user's snapshot history.
        user_id:        Target user.
        target_version: Exact version number to restore.
        reason:         Reason label for the rollback metric
                        (``"user_command"``, ``"quality_check"``, or ``"manual"``).

    Returns:
        The newly created rollback ``SnapshotRecord`` (now active).

    Raises:
        RollbackError: When *target_version* does not exist for this user.
    """
    history = await store.list_for_user(user_id, limit=50)
    target = next((s for s in history if s.version == target_version), None)
    if target is None:
        raise RollbackError(
            f"Snapshot version {target_version} not found for user {user_id}"
        )
    return await _apply_rollback(store, user_id, target, reason=reason)


async def _apply_rollback(
    store: SnapshotStore,
    user_id: uuid.UUID,
    target: SnapshotRecord,
    *,
    reason: str,
) -> SnapshotRecord:
    """Create a rollback snapshot from *target* and make it the active snapshot."""
    new_version = await store.next_version(user_id)
    rollback_snap = SnapshotRecord(
        user_id=user_id,
        version=new_version,
        system_prompt=target.system_prompt,
        skill_prompts_json=dict(target.skill_prompts_json),
        source="rollback",
    )
    await store.save(rollback_snap)
    await store.set_active(user_id, rollback_snap.id)
    PROMPT_ROLLBACKS.labels(reason=reason).inc()
    return rollback_snap
