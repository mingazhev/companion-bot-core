"""Unit tests for tdbot.prompt.schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tdbot.prompt.schemas import PromptComponents, SnapshotRecord

# ---------------------------------------------------------------------------
# PromptComponents
# ---------------------------------------------------------------------------


def test_prompt_components_minimal() -> None:
    c = PromptComponents(base_system_template="You are a helpful bot.")
    assert c.base_system_template == "You are a helpful bot."
    assert c.persona_segment == ""
    assert c.skill_packs == {}
    assert c.short_term_window == ""
    assert c.long_term_profile == ""


def test_prompt_components_full() -> None:
    c = PromptComponents(
        base_system_template="base",
        persona_segment="friendly",
        skill_packs={"math": "You can solve equations."},
        short_term_window="User asked about weather.",
        long_term_profile="User prefers short answers.",
    )
    assert c.persona_segment == "friendly"
    assert c.skill_packs["math"] == "You can solve equations."
    assert c.short_term_window == "User asked about weather."
    assert c.long_term_profile == "User prefers short answers."


def test_prompt_components_requires_base_template() -> None:
    with pytest.raises(ValidationError):
        PromptComponents()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SnapshotRecord
# ---------------------------------------------------------------------------


def test_snapshot_record_defaults() -> None:
    user_id = uuid.uuid4()
    snap = SnapshotRecord(
        user_id=user_id,
        version=1,
        system_prompt="Hello world",
        source="initial",
    )
    assert snap.user_id == user_id
    assert snap.version == 1
    assert snap.system_prompt == "Hello world"
    assert snap.source == "initial"
    assert snap.skill_prompts_json == {}
    assert isinstance(snap.id, uuid.UUID)
    assert isinstance(snap.created_at, datetime)
    assert snap.created_at.tzinfo is not None  # timezone-aware


def test_snapshot_record_created_at_utc() -> None:
    snap = SnapshotRecord(
        user_id=uuid.uuid4(),
        version=1,
        system_prompt="s",
        source="refinement",
    )
    assert snap.created_at.tzinfo == UTC


def test_snapshot_record_version_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        SnapshotRecord(
            user_id=uuid.uuid4(),
            version=0,
            system_prompt="s",
            source="initial",
        )


def test_snapshot_record_unique_ids() -> None:
    user_id = uuid.uuid4()
    snap_a = SnapshotRecord(user_id=user_id, version=1, system_prompt="a", source="initial")
    snap_b = SnapshotRecord(user_id=user_id, version=2, system_prompt="b", source="refinement")
    assert snap_a.id != snap_b.id


def test_snapshot_record_all_sources_accepted() -> None:
    user_id = uuid.uuid4()
    for v, source in enumerate(("initial", "user_command", "refinement", "rollback"), start=1):
        snap = SnapshotRecord(user_id=user_id, version=v, system_prompt="s", source=source)  # type: ignore[arg-type]
        assert snap.source == source


def test_snapshot_record_invalid_source() -> None:
    with pytest.raises(ValidationError):
        SnapshotRecord(
            user_id=uuid.uuid4(),
            version=1,
            system_prompt="s",
            source="unknown",  # type: ignore[arg-type]
        )
