"""Prompt state management: snapshot versioning, merge building, and rollback."""

from companion_bot_core.prompt.merge_builder import SECTION_SEP, build_system_prompt
from companion_bot_core.prompt.postgres_store import PostgresSnapshotStore
from companion_bot_core.prompt.rollback import (
    RollbackError,
    rollback_to_previous,
    rollback_to_version,
)
from companion_bot_core.prompt.schemas import PromptComponents, SnapshotRecord, SnapshotSource
from companion_bot_core.prompt.snapshot_store import InMemorySnapshotStore, SnapshotStore

__all__ = [
    "build_system_prompt",
    "InMemorySnapshotStore",
    "PostgresSnapshotStore",
    "SECTION_SEP",
    "PromptComponents",
    "RollbackError",
    "rollback_to_previous",
    "rollback_to_version",
    "SnapshotRecord",
    "SnapshotSource",
    "SnapshotStore",
]
