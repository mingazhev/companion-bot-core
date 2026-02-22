"""Unit tests for tdbot.refinement.schemas."""

from __future__ import annotations

import uuid

from tdbot.refinement.schemas import (
    RefinementJob,
    RefinementResult,
    RefinementRiskFlag,
    SnapshotDelta,
)

# ---------------------------------------------------------------------------
# SnapshotDelta
# ---------------------------------------------------------------------------


def test_snapshot_delta_all_none_by_default() -> None:
    delta = SnapshotDelta()
    assert delta.persona_segment is None
    assert delta.skill_packs is None
    assert delta.long_term_profile is None


def test_snapshot_delta_with_values() -> None:
    delta = SnapshotDelta(
        persona_segment="Be warm",
        skill_packs={"cooking": "You are a cooking expert"},
        long_term_profile="User likes brevity",
    )
    assert delta.persona_segment == "Be warm"
    assert delta.skill_packs == {"cooking": "You are a cooking expert"}
    assert delta.long_term_profile == "User likes brevity"


def test_snapshot_delta_partial() -> None:
    delta = SnapshotDelta(long_term_profile="User loves Python")
    assert delta.persona_segment is None
    assert delta.skill_packs is None
    assert delta.long_term_profile == "User loves Python"


# ---------------------------------------------------------------------------
# RefinementResult
# ---------------------------------------------------------------------------


def test_refinement_result_no_flags() -> None:
    result = RefinementResult(
        proposed_delta=SnapshotDelta(persona_segment="Be concise"),
        rationale="User prefers shorter replies",
    )
    assert result.risk_flags == []
    assert result.rationale == "User prefers shorter replies"


def test_refinement_result_with_flags() -> None:
    result = RefinementResult(
        proposed_delta=SnapshotDelta(),
        rationale="Injection detected",
        risk_flags=[RefinementRiskFlag.PROMPT_INJECTION],
    )
    assert RefinementRiskFlag.PROMPT_INJECTION in result.risk_flags


def test_refinement_result_model_validate_from_dict() -> None:
    data = {
        "proposed_delta": {
            "persona_segment": "Friendly",
            "skill_packs": None,
            "long_term_profile": None,
        },
        "rationale": "Observed friendliness preference",
        "risk_flags": [],
    }
    result = RefinementResult.model_validate(data)
    assert result.proposed_delta.persona_segment == "Friendly"
    assert result.risk_flags == []


def test_refinement_result_model_validate_with_risk_flags() -> None:
    data = {
        "proposed_delta": {"persona_segment": None, "skill_packs": None, "long_term_profile": None},
        "rationale": "Unsafe",
        "risk_flags": ["unsafe_role_change", "policy_violation"],
    }
    result = RefinementResult.model_validate(data)
    assert RefinementRiskFlag.UNSAFE_ROLE_CHANGE in result.risk_flags
    assert RefinementRiskFlag.POLICY_VIOLATION in result.risk_flags


# ---------------------------------------------------------------------------
# RefinementJob
# ---------------------------------------------------------------------------


def test_refinement_job_defaults() -> None:
    job = RefinementJob(user_id=uuid.uuid4())
    assert job.attempt == 0
    assert job.extra == {}
    assert job.job_id is not None
    assert job.triggered_at is not None


def test_refinement_job_with_attempt() -> None:
    user_id = uuid.uuid4()
    job = RefinementJob(user_id=user_id, attempt=2)
    assert job.attempt == 2
    assert job.user_id == user_id


# ---------------------------------------------------------------------------
# RefinementRiskFlag
# ---------------------------------------------------------------------------


def test_risk_flag_string_values() -> None:
    assert RefinementRiskFlag.PROMPT_INJECTION == "prompt_injection"
    assert RefinementRiskFlag.UNSAFE_ROLE_CHANGE == "unsafe_role_change"
    assert RefinementRiskFlag.POLICY_VIOLATION == "policy_violation"
    assert RefinementRiskFlag.SCHEMA_VIOLATION == "schema_violation"
