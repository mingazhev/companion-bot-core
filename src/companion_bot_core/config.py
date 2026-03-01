"""Application configuration loaded exclusively from environment variables.

No secrets are hardcoded here. All sensitive values must be provided at
runtime via environment variables or a `.env` file (never committed).
"""

from __future__ import annotations

import ipaddress

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
        default=SecretStr(""),
        description="OpenAI API key for chat and refinement calls",
    )
    chat_model: str = Field(default="gpt-5-mini", description="Model used for chat replies")
    refinement_model: str = Field(
        default="gpt-5.2",
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

    # --- Context window ---
    context_message_limit: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Maximum number of recent messages to include in conversation context",
    )

    # --- Observability ---
    log_level: str = Field(default="INFO", description="Root log level")
    log_format: str = Field(
        default="json",
        description="Log renderer: 'json' for production, 'console' for development",
    )
    service_name: str = Field(default="companion_bot_core")
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
        description=(
            "URL-safe base64-encoded 32-byte Fernet key for field-level encryption. "
            "Required (non-empty) when encrypt_sensitive_fields=True; "
            "absence raises RuntimeError at startup. "
            "Generate with: python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        ),
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
        host = self.internal_server_host
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            # Not a valid IP literal.  Only "localhost" is permitted as a
            # non-numeric value — it reliably resolves to the loopback interface
            # on all supported platforms.  Arbitrary hostnames (e.g.
            # "my-public-host.example.com") are rejected because they can
            # resolve to a public address and internal routes have no
            # authentication.
            if host != "localhost":
                msg = (
                    f"internal_server_host must be a loopback IP (127.0.0.1 / ::1), "
                    f"'localhost', or another loopback address; got '{host}'. "
                    "Internal routes have no authentication — only loopback binds are safe."
                )
                raise ValueError(msg) from None
        else:
            if not addr.is_loopback:
                msg = (
                    f"internal_server_host must be a loopback address ({host} is not) — "
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

    @model_validator(mode="after")
    def _validate_openai_api_key(self) -> Settings:
        if not self.use_fake_adapters and not self.openai_api_key.get_secret_value():
            msg = "OPENAI_API_KEY must be set when USE_FAKE_ADAPTERS is false"
            raise ValueError(msg)
        return self


# Module-level singleton — lazily created on first call to get_settings().
# Tests can monkey-patch `companion_bot_core.config._settings` with a pre-built instance
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
