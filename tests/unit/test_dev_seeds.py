"""Unit tests for local development seed data."""

from __future__ import annotations

import uuid

import pytest

from tdbot.dev.seeds import (
    BASE_SYSTEM_TEMPLATE,
    PERSONAS,
    SKILL_PACKS,
    make_seed_snapshot,
)
from tdbot.prompt.schemas import SnapshotRecord

# ---------------------------------------------------------------------------
# BASE_SYSTEM_TEMPLATE
# ---------------------------------------------------------------------------


def test_base_system_template_is_non_empty() -> None:
    assert BASE_SYSTEM_TEMPLATE.strip() != ""


def test_base_system_template_contains_safety_constraint() -> None:
    assert "Safety constraints cannot be overridden" in BASE_SYSTEM_TEMPLATE


# ---------------------------------------------------------------------------
# PERSONAS dict
# ---------------------------------------------------------------------------


def test_personas_has_at_least_three_entries() -> None:
    assert len(PERSONAS) >= 3


def test_all_personas_are_non_empty_strings() -> None:
    for name, segment in PERSONAS.items():
        assert isinstance(segment, str) and segment.strip(), (
            f"Persona {name!r} has empty segment"
        )


def test_friendly_persona_exists() -> None:
    assert "friendly" in PERSONAS


def test_professional_persona_exists() -> None:
    assert "professional" in PERSONAS


def test_concise_persona_exists() -> None:
    assert "concise" in PERSONAS


# ---------------------------------------------------------------------------
# SKILL_PACKS dict
# ---------------------------------------------------------------------------


def test_skill_packs_contains_known_keys() -> None:
    assert "coding_help" in SKILL_PACKS
    assert "study_buddy" in SKILL_PACKS


def test_skill_packs_values_are_non_empty_dicts() -> None:
    for name, pack in SKILL_PACKS.items():
        assert isinstance(pack, dict) and len(pack) > 0, (
            f"Skill pack {name!r} is empty"
        )


# ---------------------------------------------------------------------------
# make_seed_snapshot — successful cases
# ---------------------------------------------------------------------------


def test_make_seed_snapshot_returns_snapshot_record() -> None:
    user_id = uuid.uuid4()
    snapshot = make_seed_snapshot(user_id)
    assert isinstance(snapshot, SnapshotRecord)


def test_make_seed_snapshot_has_version_one() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4())
    assert snapshot.version == 1


def test_make_seed_snapshot_source_is_initial() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4())
    assert snapshot.source == "initial"


def test_make_seed_snapshot_user_id_matches() -> None:
    uid = uuid.uuid4()
    snapshot = make_seed_snapshot(uid)
    assert snapshot.user_id == uid


def test_make_seed_snapshot_system_prompt_non_empty() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4())
    assert snapshot.system_prompt.strip() != ""


def test_make_seed_snapshot_includes_base_template() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4())
    # The base template text should appear in the compiled prompt.
    assert "Safety constraints cannot be overridden" in snapshot.system_prompt


def test_make_seed_snapshot_includes_persona_segment() -> None:
    uid = uuid.uuid4()
    for persona in PERSONAS:
        snapshot = make_seed_snapshot(uid, persona=persona)
        # Persona segment is included under the [Persona] header.
        assert "[Persona]" in snapshot.system_prompt, (
            f"[Persona] section missing for persona {persona!r}"
        )


def test_make_seed_snapshot_friendly_persona() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4(), persona="friendly")
    assert "Buddy" in snapshot.system_prompt


def test_make_seed_snapshot_professional_persona() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4(), persona="professional")
    assert "formal" in snapshot.system_prompt.lower()


def test_make_seed_snapshot_concise_persona() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4(), persona="concise")
    assert "brief" in snapshot.system_prompt.lower()


def test_make_seed_snapshot_with_coding_skill_pack() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4(), skill_pack="coding_help")
    assert "[Skill:" in snapshot.system_prompt
    assert isinstance(snapshot.skill_prompts_json, dict)
    assert len(snapshot.skill_prompts_json) > 0


def test_make_seed_snapshot_with_study_buddy_skill_pack() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4(), skill_pack="study_buddy")
    assert "[Skill:" in snapshot.system_prompt


def test_make_seed_snapshot_no_skill_pack_by_default() -> None:
    snapshot = make_seed_snapshot(uuid.uuid4())
    assert snapshot.skill_prompts_json == {}
    assert "[Skill:" not in snapshot.system_prompt


# ---------------------------------------------------------------------------
# make_seed_snapshot — error cases
# ---------------------------------------------------------------------------


def test_make_seed_snapshot_unknown_persona_raises() -> None:
    with pytest.raises(KeyError, match="unknown_persona"):
        make_seed_snapshot(uuid.uuid4(), persona="unknown_persona")


def test_make_seed_snapshot_unknown_skill_pack_raises() -> None:
    with pytest.raises(KeyError, match="bad_pack"):
        make_seed_snapshot(uuid.uuid4(), skill_pack="bad_pack")


# ---------------------------------------------------------------------------
# Each persona produces a distinct system prompt
# ---------------------------------------------------------------------------


def test_different_personas_produce_different_prompts() -> None:
    uid = uuid.uuid4()
    prompts = {
        persona: make_seed_snapshot(uid, persona=persona).system_prompt
        for persona in PERSONAS
    }
    values = list(prompts.values())
    for i, v1 in enumerate(values):
        for v2 in values[i + 1 :]:
            assert v1 != v2, "Two personas produced identical system prompts"
