"""Deterministic quality checks for persona test scenarios.

Wraps functions from :mod:`companion_bot_core.quality.checks` into a
single ``run_checks`` entry point used by the scenario runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from companion_bot_core.quality.checks import (
    contains_name,
    count_sentences,
    has_ai_markers,
    has_menu_pattern,
    is_short_farewell,
)


@dataclass
class CheckResult:
    """Outcome of a single named check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class MessageCheckReport:
    """Aggregated check report for one assistant response."""

    user_message: str
    assistant_response: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


def run_checks(
    user_message: str,
    response: str,
    checks_config: dict[str, Any],
) -> MessageCheckReport:
    """Apply deterministic checks described in *checks_config* to *response*.

    Supported keys (matching YAML ``checks:`` block):

    - ``no_ai_markers``  (bool)  — response must contain no AI self-reference
    - ``no_menu``        (bool)  — response must not look like a numbered menu
    - ``max_sentences``  (int)   — response must have at most N sentences
    - ``is_short_farewell`` (bool) — response should be a brief farewell
    - ``contains_name``  (str)   — response must include the given name
    """
    report = MessageCheckReport(
        user_message=user_message,
        assistant_response=response,
    )

    if checks_config.get("no_ai_markers"):
        markers = has_ai_markers(response)
        report.results.append(
            CheckResult(
                name="no_ai_markers",
                passed=len(markers) == 0,
                detail=f"found: {markers}" if markers else "",
            )
        )

    if checks_config.get("no_menu"):
        menu = has_menu_pattern(response)
        report.results.append(
            CheckResult(
                name="no_menu",
                passed=not menu,
                detail="menu pattern detected" if menu else "",
            )
        )

    max_sent = checks_config.get("max_sentences")
    if max_sent is not None:
        n = count_sentences(response)
        report.results.append(
            CheckResult(
                name="max_sentences",
                passed=n <= max_sent,
                detail=f"sentences={n}, max={max_sent}",
            )
        )

    if checks_config.get("is_short_farewell"):
        ok = is_short_farewell(response)
        report.results.append(
            CheckResult(
                name="is_short_farewell",
                passed=ok,
                detail="" if ok else "not a short farewell",
            )
        )

    name = checks_config.get("contains_name")
    if name:
        ok = contains_name(response, name)
        report.results.append(
            CheckResult(
                name="contains_name",
                passed=ok,
                detail="" if ok else f"'{name}' not found",
            )
        )

    return report
