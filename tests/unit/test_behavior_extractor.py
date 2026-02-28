"""Unit tests for companion_bot_core.behavior.extractor (parameter extraction)."""

from __future__ import annotations

import pytest

from companion_bot_core.behavior.extractor import (
    extract_persona_name,
    extract_skill_topic,
    extract_tone,
)

# ---------------------------------------------------------------------------
# extract_tone
# ---------------------------------------------------------------------------


class TestExtractTone:
    """Tests for extracting tones from natural language messages."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Be more friendly please", "friendly"),
            ("Sound more professional", "professional"),
            ("Be more playful in your responses", "playful"),
            ("Respond in a more casual way", "casual"),
            ("Be neutral", "neutral"),
        ],
    )
    def test_extracts_canonical_tones(self, text: str, expected: str) -> None:
        assert extract_tone(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Be warmer in your responses", "friendly"),
            ("Sound more formal please", "professional"),
            ("Be more fun", "playful"),
            ("Talk in a relaxed way", "casual"),
            ("Be more informal", "casual"),
            ("Sound more serious", "professional"),
            ("Be humorous", "playful"),
        ],
    )
    def test_extracts_alias_tones(self, text: str, expected: str) -> None:
        assert extract_tone(text) == expected

    def test_returns_none_for_unrecognised_tone(self) -> None:
        assert extract_tone("Tell me about the weather") is None

    def test_returns_none_for_empty_string(self) -> None:
        assert extract_tone("") is None

    def test_case_insensitive(self) -> None:
        assert extract_tone("Be more FRIENDLY") == "friendly"


# ---------------------------------------------------------------------------
# extract_persona_name
# ---------------------------------------------------------------------------


class TestExtractPersonaName:
    """Tests for extracting persona names from natural language messages."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("You are now Alex", "Alex"),
            ("Your name is now Bob", "Bob"),
            ("Call yourself Charlie", "Charlie"),
            ("From now on you are Diana", "Diana"),
            ("From now on you're Eve", "Eve"),
            ("Pretend to be Frank", "Frank"),
            ("Roleplay as Grace", "Grace"),
            ("Act as a Wizard", "Wizard"),
            ("Play the role of Max", "Max"),
            ("Imagine you are Sophie", "Sophie"),
            ("Change your persona to Luna", "Luna"),
        ],
    )
    def test_extracts_persona_name(self, text: str, expected: str) -> None:
        assert extract_persona_name(text) == expected

    def test_returns_none_for_normal_message(self) -> None:
        assert extract_persona_name("What is the weather today?") is None

    def test_returns_none_for_empty_string(self) -> None:
        assert extract_persona_name("") is None

    def test_strips_trailing_punctuation(self) -> None:
        assert extract_persona_name("You are now Alex!") == "Alex"

    def test_multi_word_name(self) -> None:
        result = extract_persona_name("You are now Captain Jack")
        assert result == "Captain Jack"

    def test_exactly_64_char_name_accepted(self) -> None:
        """A name of exactly 64 characters (the limit) must be extracted."""
        name = "A" + "a" * 63  # 1 + 63 = 64 characters
        assert len(name) == 64
        result = extract_persona_name(f"You are now {name}")
        assert result == name

    def test_name_over_64_chars_rejected(self) -> None:
        """A name exceeding 64 characters must not be extracted."""
        name = "A" + "a" * 64  # 65 characters — regex captures at most 64
        result = extract_persona_name(f"You are now {name}")
        # The regex captures at most 64 chars total, so the full 65-char name
        # is never returned (either truncated or not matched due to len guard).
        assert result != name


# ---------------------------------------------------------------------------
# extract_skill_topic
# ---------------------------------------------------------------------------


class TestExtractSkillTopic:
    """Tests for extracting skill topics from natural language messages."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Add skill for cooking", "cooking"),
            ("Remove skill for math", "math"),
            ("Help me with cooking from now on", "cooking"),
            ("Stop helping me with finance", "finance"),
            ("Learn about photography", "photography"),
        ],
    )
    def test_extracts_skill_topic(self, text: str, expected: str) -> None:
        assert extract_skill_topic(text) == expected

    def test_returns_none_for_normal_message(self) -> None:
        assert extract_skill_topic("What is the weather today?") is None

    def test_returns_none_for_empty_string(self) -> None:
        assert extract_skill_topic("") is None
