"""Unit tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from companion_bot_core.config import Settings, get_settings


def _make_settings(**kwargs: object) -> Settings:
    """Construct a Settings with required fields plus any overrides."""
    defaults: dict[str, object] = {
        "telegram_bot_token": SecretStr("tok"),
        "database_url": SecretStr("postgresql+asyncpg://u:p@h/db"),
        "redis_url": SecretStr("redis://localhost:6379/0"),
        "openai_api_key": SecretStr("sk-x"),
    }
    defaults.update(kwargs)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestSettingsDefaults:
    def test_get_settings_returns_singleton(self, test_settings: Settings) -> None:
        # After monkeypatching, get_settings() should return the test instance.
        assert get_settings() is test_settings

    def test_chat_model_default(self) -> None:
        assert _make_settings().chat_model == "gpt-5-mini"

    def test_refinement_model_default(self) -> None:
        assert _make_settings().refinement_model == "gpt-5.2"

    def test_conversation_ttl_default_is_7_days(self) -> None:
        assert _make_settings().conversation_ttl_seconds == 7 * 24 * 3600

    def test_secrets_are_not_exposed_in_repr(self) -> None:
        s = _make_settings(
            telegram_bot_token=SecretStr("super-secret-token"),
            openai_api_key=SecretStr("sk-secret"),
        )
        repr_str = repr(s)
        assert "super-secret-token" not in repr_str
        assert "sk-secret" not in repr_str

    def test_required_fields_missing_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_environment_tag(self, test_settings: Settings) -> None:
        assert test_settings.environment == "test"

    def test_service_name_default(self) -> None:
        assert _make_settings().service_name == "companion_bot_core"

    def test_pool_bounds(self) -> None:
        s = _make_settings(database_pool_min=3, database_pool_max=20)
        assert s.database_pool_min == 3
        assert s.database_pool_max == 20


class TestInternalHostValidation:
    def test_loopback_ipv4_accepted(self) -> None:
        s = _make_settings(internal_server_host="127.0.0.1")
        assert s.internal_server_host == "127.0.0.1"

    def test_loopback_ipv6_accepted(self) -> None:
        s = _make_settings(internal_server_host="::1")
        assert s.internal_server_host == "::1"

    def test_hostname_accepted(self) -> None:
        # Non-IP-literal hostnames pass validation (assumed loopback by deployer).
        s = _make_settings(internal_server_host="localhost")
        assert s.internal_server_host == "localhost"

    def test_wildcard_ipv4_rejected(self) -> None:
        with pytest.raises(ValidationError, match="loopback"):
            _make_settings(internal_server_host="0.0.0.0")

    def test_wildcard_ipv6_rejected(self) -> None:
        with pytest.raises(ValidationError, match="loopback"):
            _make_settings(internal_server_host="::")

    def test_routable_ipv4_rejected(self) -> None:
        with pytest.raises(ValidationError, match="loopback"):
            _make_settings(internal_server_host="192.168.1.100")

    def test_private_rfc1918_rejected(self) -> None:
        with pytest.raises(ValidationError, match="loopback"):
            _make_settings(internal_server_host="10.0.0.1")

    def test_arbitrary_hostname_rejected(self) -> None:
        # A public or private hostname must be rejected because it could
        # resolve to a non-loopback address; internal routes have no auth.
        with pytest.raises(ValidationError):
            _make_settings(internal_server_host="my-public-host.example.com")

    def test_internal_hostname_rejected(self) -> None:
        # Even internal-sounding hostnames are rejected (cannot verify resolution).
        with pytest.raises(ValidationError):
            _make_settings(internal_server_host="internal-service")
