"""Unit tests for the persona test framework (checks, runner, judge)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from tests.persona.checks import CheckResult, MessageCheckReport, run_checks
from tests.persona.judge import JudgeScore, _format_dialogue, _parse_judge_response
from tests.persona.runner import (
    DialogueTurn,
    ScenarioResult,
    _load_scenario,
    generate_report,
)

# -----------------------------------------------------------------------
# checks.py
# -----------------------------------------------------------------------


class TestRunChecks:
    def test_no_ai_markers_pass(self) -> None:
        report = run_checks("hi", "Привет! Как дела?", {"no_ai_markers": True})
        assert report.all_passed
        assert len(report.results) == 1
        assert report.results[0].name == "no_ai_markers"

    def test_no_ai_markers_fail(self) -> None:
        report = run_checks(
            "hi",
            "Как языковая модель я не могу это сделать",
            {"no_ai_markers": True},
        )
        assert not report.all_passed
        assert "Как языковая модель" in report.results[0].detail

    def test_no_menu_pass(self) -> None:
        report = run_checks("hi", "Понимаю тебя. Это непросто.", {"no_menu": True})
        assert report.all_passed

    def test_no_menu_fail(self) -> None:
        text = "Вот что можно сделать:\n1. Первое\n2. Второе\n3. Третье"
        report = run_checks("hi", text, {"no_menu": True})
        assert not report.all_passed

    def test_max_sentences_pass(self) -> None:
        report = run_checks("hi", "Одно. Два. Три.", {"max_sentences": 5})
        assert report.all_passed
        assert "sentences=3" in report.results[0].detail

    def test_max_sentences_fail(self) -> None:
        text = "Раз. Два. Три. Четыре. Пять. Шесть."
        report = run_checks("hi", text, {"max_sentences": 3})
        assert not report.all_passed

    def test_is_short_farewell_pass(self) -> None:
        report = run_checks("пока", "Пока! Удачи.", {"is_short_farewell": True})
        assert report.all_passed

    def test_is_short_farewell_fail(self) -> None:
        report = run_checks("пока", "Это длинный ответ без прощания.", {"is_short_farewell": True})
        assert not report.all_passed

    def test_contains_name_pass(self) -> None:
        report = run_checks("меня зовут Маша", "Привет, Маша!", {"contains_name": "Маша"})
        assert report.all_passed

    def test_contains_name_fail(self) -> None:
        report = run_checks("меня зовут Маша", "Привет!", {"contains_name": "Маша"})
        assert not report.all_passed

    def test_multiple_checks(self) -> None:
        report = run_checks(
            "hi",
            "Понимаю.",
            {"no_ai_markers": True, "no_menu": True, "max_sentences": 3},
        )
        assert report.all_passed
        assert len(report.results) == 3

    def test_empty_checks_config(self) -> None:
        report = run_checks("hi", "Hello", {})
        assert report.all_passed
        assert len(report.results) == 0

    def test_report_properties(self) -> None:
        report = MessageCheckReport(
            user_message="hi",
            assistant_response="hello",
            results=[
                CheckResult(name="a", passed=True),
                CheckResult(name="b", passed=False, detail="fail"),
            ],
        )
        assert not report.all_passed


# -----------------------------------------------------------------------
# runner.py — _load_scenario
# -----------------------------------------------------------------------


class TestLoadScenario:
    def test_valid_scenario(self, tmp_path: Path) -> None:
        data = {
            "scenario": {"name": "test", "persona": "friendly"},
            "messages": [{"user": "hello", "checks": {"no_ai_markers": True}}],
        }
        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump(data))
        loaded = _load_scenario(f)
        assert loaded["scenario"]["name"] == "test"

    def test_missing_scenario_key(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump({"messages": [{"user": "hi"}]}))
        with pytest.raises(ValueError, match="Missing 'scenario'"):
            _load_scenario(f)

    def test_missing_name(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump({"scenario": {"persona": "x"}, "messages": [{"user": "hi"}]}))
        with pytest.raises(ValueError, match="Missing 'scenario.name'"):
            _load_scenario(f)

    def test_empty_messages(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump({"scenario": {"name": "t"}, "messages": []}))
        with pytest.raises(ValueError, match="empty 'messages'"):
            _load_scenario(f)


# -----------------------------------------------------------------------
# runner.py — ScenarioResult / report generation
# -----------------------------------------------------------------------


class TestScenarioResult:
    def test_all_checks_passed(self) -> None:
        report = MessageCheckReport(
            user_message="hi",
            assistant_response="hello",
            results=[CheckResult(name="a", passed=True)],
        )
        result = ScenarioResult(
            name="test",
            description="desc",
            persona="friendly",
            skill_pack=None,
            turns=[DialogueTurn("hi", "hello", report)],
        )
        assert result.all_checks_passed
        assert result.total_checks == 1
        assert result.passed_checks == 1

    def test_some_checks_failed(self) -> None:
        report = MessageCheckReport(
            user_message="hi",
            assistant_response="hello",
            results=[
                CheckResult(name="a", passed=True),
                CheckResult(name="b", passed=False),
            ],
        )
        result = ScenarioResult(
            name="test",
            description="",
            persona="friendly",
            skill_pack=None,
            turns=[DialogueTurn("hi", "hello", report)],
        )
        assert not result.all_checks_passed
        assert result.total_checks == 2
        assert result.passed_checks == 1


class TestGenerateReport:
    def test_report_contains_key_info(self) -> None:
        report = MessageCheckReport(
            user_message="Привет",
            assistant_response="Привет!",
            results=[CheckResult(name="no_ai_markers", passed=True)],
        )
        result = ScenarioResult(
            name="test_scenario",
            description="A test",
            persona="friendly",
            skill_pack="coding_help",
            turns=[DialogueTurn("Привет", "Привет!", report)],
        )
        md = generate_report(result)
        assert "test_scenario" in md
        assert "friendly" in md
        assert "coding_help" in md
        assert "PASS" in md
        assert "1/1" in md

    def test_report_shows_failures(self) -> None:
        report = MessageCheckReport(
            user_message="hi",
            assistant_response="As an AI I cannot...",
            results=[CheckResult(name="no_ai_markers", passed=False, detail="found AI marker")],
        )
        result = ScenarioResult(
            name="fail_test",
            description="",
            persona="friendly",
            skill_pack=None,
            turns=[DialogueTurn("hi", "As an AI I cannot...", report)],
        )
        md = generate_report(result)
        assert "FAIL" in md
        assert "0/1" in md


# -----------------------------------------------------------------------
# judge.py — parsing
# -----------------------------------------------------------------------


class TestJudgeParsing:
    def test_parse_valid_json(self) -> None:
        raw = json.dumps(
            {
                "empathy": 4,
                "naturalness": 5,
                "relevance": 4,
                "brevity": 3,
                "safety": 5,
                "overall": 4.2,
                "notes": "Good dialogue",
            }
        )
        score = _parse_judge_response(raw)
        assert score.empathy == 4
        assert score.naturalness == 5
        assert score.overall == 4.2
        assert score.notes == "Good dialogue"

    def test_parse_json_with_code_fences(self) -> None:
        data = {
            "empathy": 3, "naturalness": 3, "relevance": 3,
            "brevity": 3, "safety": 3, "overall": 3.0, "notes": "",
        }
        raw = f"```json\n{json.dumps(data)}\n```"
        score = _parse_judge_response(raw)
        assert score.empathy == 3

    def test_parse_invalid_score_raises(self) -> None:
        data = {
            "empathy": 6, "naturalness": 3, "relevance": 3,
            "brevity": 3, "safety": 3,
        }
        raw = json.dumps(data)
        with pytest.raises(ValueError, match="Invalid score"):
            _parse_judge_response(raw)

    def test_parse_missing_dimension_raises(self) -> None:
        raw = json.dumps({"empathy": 3, "naturalness": 3})
        with pytest.raises(ValueError, match="Invalid score"):
            _parse_judge_response(raw)

    def test_overall_defaults_to_average(self) -> None:
        raw = json.dumps(
            {"empathy": 4, "naturalness": 4, "relevance": 4, "brevity": 4, "safety": 4, "notes": ""}
        )
        score = _parse_judge_response(raw)
        assert score.overall == 4.0

    def test_dimension_scores_property(self) -> None:
        score = JudgeScore(
            empathy=3, naturalness=4, relevance=5, brevity=2, safety=5,
            overall=3.8, notes="",
        )
        dims = score.dimension_scores
        assert dims["empathy"] == 3
        assert len(dims) == 5


class TestFormatDialogue:
    def test_format(self) -> None:
        turns = [
            DialogueTurn("Привет", "Привет!"),
            DialogueTurn("Как дела?", "Хорошо!"),
        ]
        text = _format_dialogue(turns)
        assert "Turn 1:" in text
        assert "Turn 2:" in text
        assert "Привет" in text
        assert "Хорошо!" in text


# -----------------------------------------------------------------------
# YAML scenario files — structural validation
# -----------------------------------------------------------------------


_SCENARIO_DIR = Path(__file__).resolve().parent.parent / "persona" / "scenarios"


@pytest.mark.parametrize(
    "scenario_file",
    sorted(_SCENARIO_DIR.glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_scenario_yaml_is_valid(scenario_file: Path) -> None:
    """Every YAML scenario file has valid structure."""
    data = _load_scenario(scenario_file)
    scenario = data["scenario"]
    assert "name" in scenario
    assert "persona" in scenario
    for msg in data["messages"]:
        assert "user" in msg, f"Missing 'user' key in message: {msg}"


@pytest.mark.parametrize(
    "scenario_file",
    sorted(_SCENARIO_DIR.glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_scenario_has_at_least_3_messages(scenario_file: Path) -> None:
    """Each scenario should have at least 3 message turns."""
    data = _load_scenario(scenario_file)
    assert len(data["messages"]) >= 3
