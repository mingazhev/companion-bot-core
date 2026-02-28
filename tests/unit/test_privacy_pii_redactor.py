"""Unit tests for the PII redaction structlog processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from companion_bot_core.privacy.pii_redactor import _PII_FIELDS, _REDACTED, redact_pii

if TYPE_CHECKING:
    from collections.abc import MutableMapping


class TestRedactPii:
    def _call(self, event_dict: dict[str, Any]) -> MutableMapping[str, Any]:
        """Invoke redact_pii with dummy logger/method args."""
        return redact_pii(None, "info", event_dict)

    # --- Fields that MUST be redacted ---

    def test_redacts_content(self) -> None:
        result = self._call({"event": "msg", "content": "hello world"})
        assert result["content"] == _REDACTED

    def test_redacts_message_text(self) -> None:
        result = self._call({"event": "msg", "message_text": "secret text"})
        assert result["message_text"] == _REDACTED

    def test_redacts_system_prompt(self) -> None:
        result = self._call({"event": "snap", "system_prompt": "you are a pirate"})
        assert result["system_prompt"] == _REDACTED

    def test_redacts_skill_prompts_json(self) -> None:
        result = self._call({"event": "snap", "skill_prompts_json": {"cook": "describe recipes"}})
        assert result["skill_prompts_json"] == _REDACTED

    def test_redacts_style_constraints(self) -> None:
        result = self._call({"event": "profile", "style_constraints": "never use jargon"})
        assert result["style_constraints"] == _REDACTED

    def test_redacts_reply(self) -> None:
        result = self._call({"event": "chat", "reply": "sure, I can help!"})
        assert result["reply"] == _REDACTED

    def test_redacts_persona_name(self) -> None:
        result = self._call({"event": "profile", "persona_name": "Alex"})
        assert result["persona_name"] == _REDACTED

    # --- Fields that must NOT be redacted ---

    def test_preserves_event(self) -> None:
        result = self._call({"event": "user_provisioned", "content": "x"})
        assert result["event"] == "user_provisioned"

    def test_preserves_internal_user_id(self) -> None:
        uid = "abc-123"
        result = self._call({"event": "e", "internal_user_id": uid})
        assert result["internal_user_id"] == uid

    def test_preserves_correlation_id(self) -> None:
        cid = "deadbeef"
        result = self._call({"event": "e", "correlation_id": cid})
        assert result["correlation_id"] == cid

    def test_preserves_elapsed_ms(self) -> None:
        result = self._call({"event": "e", "elapsed_ms": 42.1})
        assert result["elapsed_ms"] == 42.1

    def test_no_pii_fields_unchanged(self) -> None:
        event_dict = {"event": "hello", "status": "ok", "count": 5}
        result = self._call(event_dict)
        assert result == {"event": "hello", "status": "ok", "count": 5}

    # --- Edge cases ---

    def test_empty_dict(self) -> None:
        result = self._call({})
        assert result == {}

    def test_pii_value_none_still_redacted(self) -> None:
        # None is a valid dict value; we still replace it with [REDACTED]
        result = self._call({"content": None})
        assert result["content"] == _REDACTED

    def test_multiple_pii_fields_all_redacted(self) -> None:
        event_dict = {
            "event": "e",
            "content": "sensitive",
            "reply": "also sensitive",
            "persona_name": "Bob",
            "safe_field": "keep me",
        }
        result = self._call(event_dict)
        assert result["content"] == _REDACTED
        assert result["reply"] == _REDACTED
        assert result["persona_name"] == _REDACTED
        assert result["safe_field"] == "keep me"
        assert result["event"] == "e"

    def test_returns_same_dict_object(self) -> None:
        event_dict: dict[str, str] = {"event": "e"}
        result = self._call(event_dict)
        assert result is event_dict

    def test_pii_fields_set_is_non_empty(self) -> None:
        assert len(_PII_FIELDS) > 0

    def test_redacted_sentinel_is_string(self) -> None:
        assert isinstance(_REDACTED, str)
        assert _REDACTED  # non-empty
