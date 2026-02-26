"""Unit tests for tdbot.prompt.merge_builder.build_system_prompt."""

from __future__ import annotations

from tdbot.prompt.merge_builder import SECTION_SEP, build_system_prompt
from tdbot.prompt.schemas import PromptComponents

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_only(template: str = "You are a companion bot.") -> PromptComponents:
    return PromptComponents(base_system_template=template)


# ---------------------------------------------------------------------------
# Base template
# ---------------------------------------------------------------------------


def test_base_only_returns_stripped_template() -> None:
    result = build_system_prompt(_base_only("  Hello world  "))
    assert result == "Hello world"


def test_base_template_always_present() -> None:
    result = build_system_prompt(_base_only("core"))
    assert result.startswith("core")


# ---------------------------------------------------------------------------
# Persona segment
# ---------------------------------------------------------------------------


def test_persona_segment_appended_with_header() -> None:
    c = PromptComponents(
        base_system_template="base",
        persona_segment="Friendly and concise.",
    )
    result = build_system_prompt(c)
    assert "[Persona]\nFriendly and concise." in result


def test_blank_persona_segment_omitted() -> None:
    c = PromptComponents(base_system_template="base", persona_segment="   ")
    result = build_system_prompt(c)
    assert "[Persona]" not in result


# ---------------------------------------------------------------------------
# Skill packs
# ---------------------------------------------------------------------------


def test_single_skill_appended_with_header() -> None:
    c = PromptComponents(
        base_system_template="base",
        skill_packs={"math": "Solve equations step by step."},
    )
    result = build_system_prompt(c)
    assert "[Skill: math]\nSolve equations step by step." in result


def test_multiple_skills_sorted_alphabetically() -> None:
    c = PromptComponents(
        base_system_template="base",
        skill_packs={"writing": "Write creatively.", "math": "Do maths."},
    )
    result = build_system_prompt(c)
    math_pos = result.index("[Skill: math]")
    writing_pos = result.index("[Skill: writing]")
    assert math_pos < writing_pos


def test_empty_skill_prompt_omitted() -> None:
    c = PromptComponents(
        base_system_template="base",
        skill_packs={"empty": "   ", "real": "Do something."},
    )
    result = build_system_prompt(c)
    assert "[Skill: empty]" not in result
    assert "[Skill: real]" in result


def test_no_skills_no_skill_headers() -> None:
    result = build_system_prompt(_base_only())
    assert "[Skill:" not in result


# ---------------------------------------------------------------------------
# Long-term profile
# ---------------------------------------------------------------------------


def test_long_term_profile_appended_with_header() -> None:
    c = PromptComponents(
        base_system_template="base",
        long_term_profile="User prefers bullet points.",
    )
    result = build_system_prompt(c)
    assert "[Long-term Profile]\nUser prefers bullet points." in result


def test_blank_long_term_profile_omitted() -> None:
    c = PromptComponents(base_system_template="base", long_term_profile="")
    result = build_system_prompt(c)
    assert "[Long-term Profile]" not in result


# ---------------------------------------------------------------------------
# Short-term window
# ---------------------------------------------------------------------------


def test_short_term_window_appended_with_header() -> None:
    c = PromptComponents(
        base_system_template="base",
        short_term_window="User asked about cats.",
    )
    result = build_system_prompt(c)
    assert "[Recent Context]\nUser asked about cats." in result


def test_blank_short_term_window_omitted() -> None:
    c = PromptComponents(base_system_template="base", short_term_window="")
    result = build_system_prompt(c)
    assert "[Recent Context]" not in result


# ---------------------------------------------------------------------------
# Section ordering
# ---------------------------------------------------------------------------


def test_section_order_base_persona_skill_longterm_shortterm() -> None:
    c = PromptComponents(
        base_system_template="base",
        persona_segment="persona",
        skill_packs={"a": "skill-a"},
        long_term_profile="long-term",
        short_term_window="short-term",
    )
    result = build_system_prompt(c)
    positions = {
        "base": result.index("base"),
        "persona": result.index("[Persona]"),
        "skill": result.index("[Skill: a]"),
        "long_term": result.index("[Long-term Profile]"),
        "short_term": result.index("[Recent Context]"),
    }
    assert (
        positions["base"]
        < positions["persona"]
        < positions["skill"]
        < positions["long_term"]
        < positions["short_term"]
    )


def test_sections_separated_by_separator() -> None:
    c = PromptComponents(
        base_system_template="base",
        persona_segment="persona",
    )
    result = build_system_prompt(c)
    assert SECTION_SEP in result


def test_single_section_no_separator() -> None:
    result = build_system_prompt(_base_only("only-base"))
    assert SECTION_SEP not in result


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_build_is_deterministic() -> None:
    c = PromptComponents(
        base_system_template="base",
        persona_segment="p",
        skill_packs={"z": "z-prompt", "a": "a-prompt"},
        long_term_profile="lt",
        short_term_window="st",
    )
    assert build_system_prompt(c) == build_system_prompt(c)
