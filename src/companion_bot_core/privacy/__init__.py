"""Privacy and data-control utilities.

Modules
-------
pii_redactor      structlog processor that redacts PII fields before log output.
field_encryption  Fernet-based symmetric field-level encryption for at-rest data.
ttl_sweeper       Batch-delete conversation_messages whose TTL has expired.
delete_user       Hard-delete all personal data for a user on request.
"""

from __future__ import annotations

from companion_bot_core.privacy.delete_user import hard_delete_user
from companion_bot_core.privacy.field_encryption import FieldEncryptor
from companion_bot_core.privacy.pii_redactor import redact_pii
from companion_bot_core.privacy.ttl_sweeper import sweep_expired_messages

__all__ = [
    "FieldEncryptor",
    "hard_delete_user",
    "redact_pii",
    "sweep_expired_messages",
]
