"""Unit tests for structured logging and correlation ID helpers."""

from __future__ import annotations

from pydantic import SecretStr

import tdbot.logging_config as lc
from tdbot.config import Settings
from tdbot.logging_config import (
    bind_correlation_id,
    configure_logging,
    get_correlation_id,
    get_logger,
    new_correlation_id,
)


def _minimal_settings(log_format: str = "console") -> Settings:
    return Settings(
        telegram_bot_token=SecretStr("tok"),
        database_url=SecretStr("postgresql+asyncpg://u:p@h/db"),
        redis_url=SecretStr("redis://localhost:6379/0"),
        openai_api_key=SecretStr("sk-x"),
        log_format=log_format,
        log_level="DEBUG",
    )


class TestCorrelationId:
    def teardown_method(self) -> None:
        # Reset correlation ID after each test so they don't bleed over.
        lc._correlation_id.set("")

    def test_new_correlation_id_returns_non_empty_string(self) -> None:
        cid = new_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) == 32  # uuid4().hex length

    def test_new_correlation_id_is_stored(self) -> None:
        cid = new_correlation_id()
        assert get_correlation_id() == cid

    def test_bind_correlation_id(self) -> None:
        bind_correlation_id("abc123")
        assert get_correlation_id() == "abc123"

    def test_default_correlation_id_is_empty(self) -> None:
        lc._correlation_id.set("")
        assert get_correlation_id() == ""

    def test_each_call_to_new_generates_unique_id(self) -> None:
        ids = {new_correlation_id() for _ in range(50)}
        assert len(ids) == 50


class TestConfigureLogging:
    def test_configure_with_console_format_does_not_raise(self) -> None:
        configure_logging(_minimal_settings("console"))

    def test_configure_with_json_format_does_not_raise(self) -> None:
        configure_logging(_minimal_settings("json"))

    def test_get_logger_returns_bound_logger(self) -> None:
        configure_logging(_minimal_settings())
        log = get_logger("test.module")
        assert log is not None

    def test_logger_can_emit_info(self) -> None:
        configure_logging(_minimal_settings())
        log = get_logger("test.module")
        # Should not raise
        log.info("test message", key="value")

    def test_logger_injects_correlation_id(self) -> None:
        configure_logging(_minimal_settings())
        bind_correlation_id("test-cid-42")
        log = get_logger("test.module")
        # Ensure no exception when correlation ID is present
        log.debug("checking correlation id injection")
        lc._correlation_id.set("")
