"""Prompt state management: snapshot versioning, merge building, and rollback."""

from tdbot.prompt.merge_builder import SECTION_SEP, build_system_prompt
from tdbot.prompt.postgres_store import PostgresSnapshotStore
from tdbot.prompt.rollback import RollbackError, rollback_to_previous, rollback_to_version
from tdbot.prompt.schemas import PromptComponents, SnapshotRecord, SnapshotSource
from tdbot.prompt.snapshot_store import InMemorySnapshotStore, SnapshotStore

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
