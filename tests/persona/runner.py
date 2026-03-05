"""Persona scenario runner.

Loads YAML scenario files, executes them through the orchestrator pipeline,
applies deterministic checks, and optionally runs the LLM judge.

Can be run as a module::

    python -m tests.persona.runner tests/persona/scenarios/katya_empathy.yaml

Or invoked programmatically from a Claude Code session::

    from tests.persona.runner import run_scenario
    result = await run_scenario("tests/persona/scenarios/katya_empathy.yaml")
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import yaml  # type: ignore[import-untyped]

from companion_bot_core.dev.seeds import make_seed_snapshot
from companion_bot_core.orchestrator.orchestrator import process_message
from companion_bot_core.prompt.snapshot_store import InMemorySnapshotStore

from .checks import MessageCheckReport, run_checks

try:
    import fakeredis.aioredis as fakeredis
except ImportError:  # pragma: no cover
    fakeredis = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DialogueTurn:
    """One user/assistant exchange with check results."""

    user_message: str
    assistant_response: str
    check_report: MessageCheckReport | None = None


@dataclass
class ScenarioResult:
    """Full result of a single scenario run."""

    name: str
    description: str
    persona: str
    skill_pack: str | None
    turns: list[DialogueTurn] = field(default_factory=list)
    error: str | None = None

    @property
    def all_checks_passed(self) -> bool:
        return all(
            t.check_report.all_passed
            for t in self.turns
            if t.check_report is not None
        )

    @property
    def total_checks(self) -> int:
        return sum(
            len(t.check_report.results)
            for t in self.turns
            if t.check_report is not None
        )

    @property
    def passed_checks(self) -> int:
        return sum(
            sum(1 for r in t.check_report.results if r.passed)
            for t in self.turns
            if t.check_report is not None
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_scenario(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML scenario file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    scenario = data.get("scenario")
    if scenario is None:
        raise ValueError(f"Missing 'scenario' key in {path}")
    if "name" not in scenario:
        raise ValueError(f"Missing 'scenario.name' in {path}")
    messages = data.get("messages")
    if not messages:
        raise ValueError(f"Missing or empty 'messages' in {path}")
    return dict(data)


def _make_mock_client(reply_text: str = "Test reply.") -> AsyncMock:
    """Create a mock ChatAPIClient returning a canned response."""
    from companion_bot_core.inference.schemas import OpenAIResponse

    response = OpenAIResponse.model_validate(
        {
            "id": "persona-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_text, "refusal": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
    )
    client = AsyncMock()
    client.chat_completion = AsyncMock(return_value=response)
    return client


def _make_db_session() -> AsyncMock:
    """Create a mock DB session with empty history."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


async def run_scenario(
    path: str | Path,
    chat_client: Any | None = None,
) -> ScenarioResult:
    """Execute a persona test scenario.

    Args:
        path: Path to YAML scenario file.
        chat_client: Optional real ChatAPIClient for live testing.
                     When ``None``, a mock client echoing user messages is used.

    Returns:
        :class:`ScenarioResult` with per-turn check reports.
    """
    data = _load_scenario(path)
    scenario = data["scenario"]
    messages_spec = data["messages"]

    result = ScenarioResult(
        name=scenario["name"],
        description=scenario.get("description", ""),
        persona=scenario.get("persona", "friendly"),
        skill_pack=scenario.get("skill_pack"),
    )

    if fakeredis is None:
        result.error = "fakeredis not installed"
        return result

    # Set up test environment
    user_id = uuid.uuid4()
    redis = fakeredis.FakeRedis(decode_responses=True)
    store = InMemorySnapshotStore()

    # Onboard persona
    snapshot = make_seed_snapshot(
        user_id,
        persona=result.persona,
        skill_pack=result.skill_pack,
    )
    await store.save(snapshot)
    await store.set_active(user_id, snapshot.id)

    # Use mock client if none provided (deterministic mode)
    use_mock = chat_client is None
    mock_client: AsyncMock | None = _make_mock_client() if use_mock else None
    active_client: Any = mock_client if use_mock else chat_client

    for msg_spec in messages_spec:
        user_text = msg_spec["user"]
        checks_config = msg_spec.get("checks", {})

        # For mock mode, update the canned response to echo the user message
        if use_mock and mock_client is not None:
            mock_client.chat_completion.return_value = _make_mock_client(
                f"Понимаю. {user_text[:50]}"
            ).chat_completion.return_value

        reply = await process_message(
            user_id=user_id,
            message_text=user_text,
            session=_make_db_session(),
            snapshot_store=store,
            redis=redis,
            chat_client=active_client,
        )

        check_report = run_checks(user_text, reply, checks_config) if checks_config else None
        result.turns.append(
            DialogueTurn(
                user_message=user_text,
                assistant_response=reply,
                check_report=check_report,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(result: ScenarioResult) -> str:
    """Generate a markdown report from a scenario result."""
    lines = [
        f"# Persona Test Report: {result.name}",
        "",
        f"**Persona:** {result.persona}",
        f"**Skill pack:** {result.skill_pack or 'none'}",
        f"**Description:** {result.description}",
        f"**Timestamp:** {datetime.now(tz=UTC).isoformat()}",
        "",
        f"## Results: {result.passed_checks}/{result.total_checks} checks passed",
        "",
    ]

    if result.error:
        lines.append(f"**Error:** {result.error}")
        lines.append("")

    for i, turn in enumerate(result.turns, 1):
        lines.append(f"### Turn {i}")
        lines.append("")
        lines.append(f"**User:** {turn.user_message}")
        lines.append("")
        lines.append(f"**Bot:** {turn.assistant_response}")
        lines.append("")

        if turn.check_report:
            for cr in turn.check_report.results:
                status = "PASS" if cr.passed else "FAIL"
                detail = f" ({cr.detail})" if cr.detail else ""
                lines.append(f"- [{status}] {cr.name}{detail}")
            lines.append("")

    return "\n".join(lines)


def save_report(result: ScenarioResult, output_dir: str | Path = "docs/reports/auto") -> Path:
    """Save a markdown report to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{result.name}.md"
    filepath = output_path / filename
    filepath.write_text(generate_report(result))
    return filepath


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run persona test scenarios")
    parser.add_argument("scenarios", nargs="+", help="Path(s) to YAML scenario files")
    parser.add_argument("--output-dir", default="docs/reports/auto", help="Report output directory")
    parsed = parser.parse_args(args)

    for scenario_path in parsed.scenarios:
        result = await run_scenario(scenario_path)
        report_path = save_report(result, parsed.output_dir)
        status = "PASSED" if result.all_checks_passed else "FAILED"
        checks = f"{result.passed_checks}/{result.total_checks}"
        print(f"[{status}] {result.name}: {checks} checks — {report_path}")


if __name__ == "__main__":
    asyncio.run(_main())
