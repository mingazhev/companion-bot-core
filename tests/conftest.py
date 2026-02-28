"""Shared pytest fixtures for Companion Bot Core test suite."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

import companion_bot_core.config as config_module
from companion_bot_core.config import Settings


@pytest.fixture()
def test_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Return a Settings instance pre-configured for tests (no real secrets needed)."""
    settings = Settings(
        telegram_bot_token=SecretStr("1234567890:AAFakeTokenForTesting"),
        database_url=SecretStr("postgresql+asyncpg://test:test@localhost:5432/companion_bot_core_test"),
        redis_url=SecretStr("redis://localhost:6379/1"),
        openai_api_key=SecretStr("sk-test-fake-key"),
        log_format="console",
        log_level="DEBUG",
        environment="test",
        encrypt_sensitive_fields=False,
    )
    monkeypatch.setattr(config_module, "_settings", settings)
    return settings
