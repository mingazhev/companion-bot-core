"""Unit tests for the FieldEncryptor (Fernet field-level encryption)."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

from tdbot.config import Settings
from tdbot.privacy.field_encryption import FieldEncryptor


def _make_key() -> bytes:
    """Generate a fresh Fernet key for testing."""
    return Fernet.generate_key()


class TestFieldEncryptorEnabled:
    """Tests with encryption enabled and an explicit key."""

    def setup_method(self) -> None:
        self.key = _make_key()
        self.enc = FieldEncryptor(self.key, enabled=True)

    def test_is_enabled(self) -> None:
        assert self.enc.is_enabled is True

    def test_encrypt_returns_different_value(self) -> None:
        plaintext = "hello world"
        token = self.enc.encrypt(plaintext)
        assert token != plaintext

    def test_decrypt_roundtrip(self) -> None:
        plaintext = "my secret persona"
        assert self.enc.decrypt(self.enc.encrypt(plaintext)) == plaintext

    def test_encrypt_non_deterministic(self) -> None:
        # Fernet tokens include a timestamp and random IV so two calls differ.
        plaintext = "same input"
        assert self.enc.encrypt(plaintext) != self.enc.encrypt(plaintext)

    def test_decrypt_same_plaintext_both_tokens(self) -> None:
        plaintext = "same input"
        t1 = self.enc.encrypt(plaintext)
        t2 = self.enc.encrypt(plaintext)
        assert self.enc.decrypt(t1) == plaintext
        assert self.enc.decrypt(t2) == plaintext

    def test_decrypt_wrong_key_raises(self) -> None:
        other_key = _make_key()
        other_enc = FieldEncryptor(other_key, enabled=True)
        token = self.enc.encrypt("secret")
        with pytest.raises(InvalidToken):
            other_enc.decrypt(token)

    def test_decrypt_safe_wrong_key_returns_default(self) -> None:
        other_key = _make_key()
        other_enc = FieldEncryptor(other_key, enabled=True)
        token = self.enc.encrypt("secret")
        assert other_enc.decrypt_safe(token, default="fallback") == "fallback"

    def test_decrypt_safe_default_is_empty_string(self) -> None:
        other_key = _make_key()
        other_enc = FieldEncryptor(other_key, enabled=True)
        token = self.enc.encrypt("secret")
        assert other_enc.decrypt_safe(token) == ""

    def test_empty_string_roundtrip(self) -> None:
        assert self.enc.decrypt(self.enc.encrypt("")) == ""

    def test_unicode_roundtrip(self) -> None:
        text = "привет мир 🌍"
        assert self.enc.decrypt(self.enc.encrypt(text)) == text

    def test_long_text_roundtrip(self) -> None:
        text = "x" * 10_000
        assert self.enc.decrypt(self.enc.encrypt(text)) == text


class TestFieldEncryptorDisabled:
    """Tests with encryption disabled (pass-through mode)."""

    def setup_method(self) -> None:
        self.enc = FieldEncryptor(None, enabled=False)

    def test_is_not_enabled(self) -> None:
        assert self.enc.is_enabled is False

    def test_encrypt_returns_unchanged(self) -> None:
        val = "plaintext"
        assert self.enc.encrypt(val) == val

    def test_decrypt_returns_unchanged(self) -> None:
        val = "plaintext"
        assert self.enc.decrypt(val) == val

    def test_decrypt_safe_returns_unchanged(self) -> None:
        val = "not a fernet token"
        assert self.enc.decrypt_safe(val, default="d") == val

    def test_encrypt_with_key_still_disabled(self) -> None:
        key = _make_key()
        enc = FieldEncryptor(key, enabled=False)
        val = "hello"
        assert enc.encrypt(val) == val


class TestFieldEncryptorEphemeral:
    """Tests with enabled=True but no key (ephemeral auto-generated key)."""

    def test_ephemeral_key_warns(self) -> None:
        with pytest.warns(UserWarning, match="FIELD_ENCRYPTION_KEY is empty"):
            enc = FieldEncryptor(None, enabled=True)
        assert enc.is_enabled is True

    def test_ephemeral_key_roundtrip(self) -> None:
        with pytest.warns(UserWarning):
            enc = FieldEncryptor(None, enabled=True)
        plaintext = "ephemeral data"
        assert enc.decrypt(enc.encrypt(plaintext)) == plaintext


class TestFromSettings:
    """Tests for FieldEncryptor.from_settings factory."""

    def test_from_settings_disabled(self, test_settings: Settings) -> None:
        # test_settings fixture sets encrypt_sensitive_fields=False
        enc = FieldEncryptor.from_settings(test_settings)
        assert enc.is_enabled is False

    def test_from_settings_enabled_with_key(self) -> None:
        key = Fernet.generate_key().decode()
        settings = Settings(  # type: ignore[call-arg]
            telegram_bot_token=SecretStr("1234567890:AAFakeToken"),
            database_url=SecretStr("postgresql+asyncpg://u:p@h/db"),
            redis_url=SecretStr("redis://localhost/0"),
            openai_api_key=SecretStr("sk-fake"),
            encrypt_sensitive_fields=True,
            field_encryption_key=SecretStr(key),
        )
        enc = FieldEncryptor.from_settings(settings)
        assert enc.is_enabled is True
        plaintext = "test value"
        assert enc.decrypt(enc.encrypt(plaintext)) == plaintext
