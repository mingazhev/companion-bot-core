"""Application configuration loaded exclusively from environment variables.

No secrets are hardcoded here. All sensitive values must be provided at
runtime via environment variables or a `.env` file (never committed).
"""

from __future__ import annotations

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level application settings resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Telegram ---
    telegram_bot_token: SecretStr = Field(
        ...,
        description="Telegram Bot API token from @BotFather",
    )
    telegram_webhook_host: str = Field(
        default="",
        description="Public HTTPS host for webhook mode (empty → polling mode)",
    )
    telegram_webhook_path: str = Field(
        default="/webhook",
        description="URL path at which the webhook endpoint is mounted",
    )

    # --- Database (PostgreSQL) ---
    database_url: SecretStr = Field(
        ...,
        description="Async PostgreSQL DSN, e.g. postgresql+asyncpg://user:pass@host/db",
    )
    database_pool_min: int = Field(default=2, ge=1)
    database_pool_max: int = Field(default=10, ge=1)

    # --- Redis ---
    redis_url: SecretStr = Field(
        ...,
        description="Redis DSN, e.g. redis://localhost:6379/0",
    )

    # --- OpenAI / model provider ---
    openai_api_key: SecretStr = Field(
        ...,
        description="OpenAI API key for chat and refinement calls",
    )
    chat_model: str = Field(default="gpt-4o-mini", description="Model used for chat replies")
    refinement_model: str = Field(
        default="gpt-4o",
        description="Model used for async prompt refinement",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Override to route to a compatible proxy",
    )

    # --- Rate limits ---
    rate_limit_messages_per_minute: int = Field(default=20, ge=1)
    rate_limit_global_rps: int = Field(default=100, ge=1)

    # --- Retention ---
    conversation_ttl_seconds: int = Field(
        default=7 * 24 * 3600,
        description="TTL for conversation_messages rows (default 7 days)",
    )

    # --- Refinement worker ---
    refinement_cadence_seconds: int = Field(
        default=3600,
        description="How often to consider users for refinement",
    )
    refinement_activity_threshold: int = Field(
        default=5,
        gt=0,
        description="Minimum new messages since last refinement to trigger a job",
    )

    # --- Observability ---
    log_level: str = Field(default="INFO", description="Root log level")
    log_format: str = Field(
        default="json",
        description="Log renderer: 'json' for production, 'console' for development",
    )
    service_name: str = Field(default="tdbot")
    environment: str = Field(default="local", description="deployment environment tag")

    # --- Internal HTTP service ---
    internal_server_host: str = Field(
        default="127.0.0.1",
        description="Bind address for the internal HTTP service (not exposed externally)",
    )
    internal_server_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="TCP port for the internal HTTP service",
    )

    # --- Security ---
    encrypt_sensitive_fields: bool = Field(
        default=False,
        description="Encrypt user-facing PII fields at rest",
    )
    field_encryption_key: SecretStr = Field(
        default=SecretStr(""),
        description="32-byte Fernet key for field-level encryption (empty → auto-generate)",
    )

    @model_validator(mode="after")
    def _validate_pool_bounds(self) -> Settings:
        if self.database_pool_max < self.database_pool_min:
            msg = (
                f"database_pool_max ({self.database_pool_max}) must be >= "
                f"database_pool_min ({self.database_pool_min})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_internal_host(self) -> Settings:
        if self.internal_server_host in ("0.0.0.0", "::"):  # noqa: S104
            msg = (
                "internal_server_host must not be 0.0.0.0 or :: — "
                "internal routes have no authentication"
            )
            raise ValueError(msg)
        return self

    # --- Local development ---
    use_fake_adapters: bool = Field(
        default=False,
        description=(
            "Replace real model API clients with FakeChatAPIClient so the bot "
            "runs locally without valid OpenAI credentials. Never enable in production."
        ),
    )


# Module-level singleton — lazily created on first call to get_settings().
# Tests can monkey-patch `tdbot.config._settings` with a pre-built instance
# before importing application code that calls get_settings().
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings instance, creating it on first call.

    pydantic-settings resolves all required fields from environment variables
    (or a .env file) at that point.  mypy cannot see this, so the call-arg
    check for Settings() is suppressed.
    """
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
