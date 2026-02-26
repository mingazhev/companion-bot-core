"""Fernet-based symmetric field-level encryption for sensitive at-rest data.

Usage
-----
Build a :class:`FieldEncryptor` once at application startup and pass it to
any component that reads or writes sensitive database fields::

    from tdbot.privacy.field_encryption import FieldEncryptor

    encryptor = FieldEncryptor.from_settings(settings)

    # Writing to DB:
    row.persona_name = encryptor.encrypt(raw_persona_name)

    # Reading from DB:
    raw_persona_name = encryptor.decrypt(row.persona_name)

When ``encrypt_sensitive_fields=False`` in settings, both ``encrypt`` and
``decrypt`` are no-ops (return the value unchanged), so callers need no
conditional logic.

Key management
--------------
Provide a URL-safe base64-encoded 32-byte Fernet key via the
``FIELD_ENCRYPTION_KEY`` environment variable.  If the key is empty and
encryption is enabled, an ephemeral key is generated at startup (warning:
data encrypted with an ephemeral key cannot be recovered after a restart).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from cryptography.fernet import Fernet, InvalidToken

if TYPE_CHECKING:
    from tdbot.config import Settings

# Uses structlog directly (not tdbot.logging_config) to avoid a circular import:
# logging_config -> privacy.__init__ -> field_encryption -> logging_config
_log = structlog.get_logger(__name__)


class FieldEncryptor:
    """Symmetric field-level encryptor using :class:`cryptography.fernet.Fernet`.

    Args:
        key_bytes: URL-safe base64-encoded 32-byte Fernet key, or ``None``
                   to auto-generate an ephemeral key.
        enabled:   When ``False`` all operations are pass-through no-ops.
    """

    def __init__(self, key_bytes: bytes | None, *, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self._fernet: Fernet | None = None
            return

        if key_bytes:
            self._fernet = Fernet(key_bytes)
        else:
            raise RuntimeError(
                "FIELD_ENCRYPTION_KEY is required when encrypt_sensitive_fields=True. "
                "Generate a key with: python -c \"from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())\". "
                "Encrypted data cannot be recovered if the key is lost or ephemeral."
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> FieldEncryptor:
        """Construct from application :class:`~tdbot.config.Settings`."""
        raw_key = settings.field_encryption_key.get_secret_value()
        key_bytes = raw_key.encode() if raw_key else None
        return cls(key_bytes, enabled=settings.encrypt_sensitive_fields)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """True when encryption is active (key is set and enabled flag is set)."""
        return self._enabled and self._fernet is not None

    def encrypt(self, value: str) -> str:
        """Return the Fernet-encrypted, URL-safe base64 token for *value*.

        When encryption is disabled, returns *value* unchanged.
        """
        if not self._enabled or self._fernet is None:
            return value
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, value: str) -> str:
        """Decrypt a Fernet token produced by :meth:`encrypt`.

        When encryption is disabled, returns *value* unchanged.

        Raises
        ------
        cryptography.fernet.InvalidToken
            If *value* is not a valid Fernet token for the current key.
        """
        if not self._enabled or self._fernet is None:
            return value
        return self._fernet.decrypt(value.encode()).decode()

    def decrypt_safe(self, value: str, default: str = "") -> str:
        """Like :meth:`decrypt` but returns *default* on ``InvalidToken``.

        Useful for reading legacy rows that were not yet encrypted.
        """
        try:
            return self.decrypt(value)
        except InvalidToken:
            _log.warning(
                "field_decryption_failed",
                reason="InvalidToken — possible key mismatch or unencrypted legacy row",
            )
            return default


# Module-level singleton for disabled encryption (pass-through no-ops).
NOOP_ENCRYPTOR = FieldEncryptor(None, enabled=False)
