"""Unit tests for tdbot.refinement.client (refine_prompt)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from tdbot.inference.schemas import ChatMessage
from tdbot.prompt.schemas import SnapshotRecord
from tdbot.refinement.client import _build_refinement_messages, refine_prompt
from tdbot.refinement.schemas import RefinementResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(system_prompt: str = "You are a helpful companion.") -> SnapshotRecord:
    return SnapshotRecord(
        user_id=uuid.uuid4(),
        version=1,
        system_prompt=system_prompt,
        source="initial",
    )


def _make_openai_response(content: str) -> MagicMock:
    """Return a minimal OpenAIResponse-like mock."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_client(content: str) -> AsyncMock:
    client = AsyncMock()
    client.chat_completion = AsyncMock(return_value=_make_openai_response(content))
    return client


# ---------------------------------------------------------------------------
# _build_refinement_messages
# ---------------------------------------------------------------------------


def test_build_refinement_messages_includes_system_prompt() -> None:
    snapshot = _make_snapshot("Be helpful.")
    messages = _build_refinement_messages(snapshot, [])
    assert messages[0].role == "system"
    assert "prompt-refinement" in messages[0].content.lower()


def test_build_refinement_messages_user_contains_config() -> None:
    snapshot = _make_snapshot("You are a cooking assistant.")
    messages = _build_refinement_messages(snapshot, [])
    user_content = messages[1].content
    assert "You are a cooking assistant." in user_content


def test_build_refinement_messages_no_recent_context() -> None:
    snapshot = _make_snapshot()
    messages = _build_refinement_messages(snapshot, [])
    assert "(no recent messages)" in messages[1].content


def test_build_refinement_messages_includes_recent_context() -> None:
    snapshot = _make_snapshot()
    history = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi!"),
    ]
    messages = _build_refinement_messages(snapshot, history)
    user_content = messages[1].content
    assert "Hello" in user_content
    assert "Hi!" in user_content


def test_build_refinement_messages_truncates_long_context() -> None:
    snapshot = _make_snapshot()
    history = [ChatMessage(role="user", content=f"msg {i}") for i in range(50)]
    messages = _build_refinement_messages(snapshot, history)
    # Only the last 30 messages should appear; earliest message (msg 0) should not
    user_content = messages[1].content
    assert "msg 0" not in user_content
    assert "msg 49" in user_content


# ---------------------------------------------------------------------------
# refine_prompt — success cases
# ---------------------------------------------------------------------------


async def test_refine_prompt_returns_result_on_valid_json() -> None:
    payload = json.dumps(
        {
            "proposed_delta": {
                "persona_segment": "Be warm and encouraging",
                "skill_packs": None,
                "long_term_profile": None,
            },
            "rationale": "User uses warm language",
            "risk_flags": [],
        }
    )
    client = _make_client(payload)
    snapshot = _make_snapshot()
    result = await refine_prompt(client, snapshot, [])
    assert isinstance(result, RefinementResult)
    assert result.proposed_delta.persona_segment == "Be warm and encouraging"
    assert result.risk_flags == []


async def test_refine_prompt_passes_low_temperature() -> None:
    no_change_delta = {"persona_segment": None, "skill_packs": None, "long_term_profile": None}
    payload = json.dumps(
        {"proposed_delta": no_change_delta, "rationale": "No change needed", "risk_flags": []}
    )
    client = _make_client(payload)
    snapshot = _make_snapshot()
    await refine_prompt(client, snapshot, [])
    call_kwargs = client.chat_completion.call_args
    assert call_kwargs.kwargs.get("temperature") == 0.2


async def test_refine_prompt_result_with_risk_flags() -> None:
    no_change_delta = {"persona_segment": None, "skill_packs": None, "long_term_profile": None}
    payload = json.dumps(
        {
            "proposed_delta": no_change_delta,
            "rationale": "Injection detected",
            "risk_flags": ["prompt_injection"],
        }
    )
    client = _make_client(payload)
    result = await refine_prompt(client, _make_snapshot(), [])
    assert "prompt_injection" in [f.value for f in result.risk_flags]


# ---------------------------------------------------------------------------
# refine_prompt — error cases
# ---------------------------------------------------------------------------


async def test_refine_prompt_raises_on_empty_content() -> None:
    client = _make_client("")
    with pytest.raises(ValueError, match="empty content"):
        await refine_prompt(client, _make_snapshot(), [])


async def test_refine_prompt_raises_on_non_json() -> None:
    client = _make_client("This is not JSON at all.")
    with pytest.raises(ValueError, match="non-JSON"):
        await refine_prompt(client, _make_snapshot(), [])


async def test_refine_prompt_raises_on_schema_mismatch() -> None:
    # Valid JSON but missing required fields
    client = _make_client(json.dumps({"wrong_field": "value"}))
    with pytest.raises(ValueError, match="schema validation"):
        await refine_prompt(client, _make_snapshot(), [])
