"""Hard-delete flow for user personal data.

When a user requests data deletion via ``/delete_my_data``, this module
performs a hard delete that:

1. Writes a minimal audit-log entry *before* deletion (so there is a record
   that a deletion was requested and executed).
2. Deletes the :class:`~tdbot.db.models.User` row.
   All personal-data tables (``user_profiles``, ``prompt_snapshots``,
   ``conversation_messages``, ``memory_compactions``,
   ``behavior_change_events``, ``jobs``) have ``ON DELETE CASCADE`` foreign
   keys and are removed automatically.
3. The ``audit_log`` rows referencing this user have ``ON DELETE SET NULL``,
   so they are preserved with ``user_id = NULL`` — the minimal audit trail
   required for compliance.

The caller is responsible for committing the transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete

from tdbot.db.models import AuditLog, User

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession


async def hard_delete_user(
    user_id: uuid.UUID,
    session: AsyncSession,
) -> None:
    """Remove all personal data for *user_id*, preserving minimal audit trail.

    Args:
        user_id: Internal UUID of the user to delete.
        session: An active :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
                 The caller is responsible for committing the transaction.

    The function is idempotent: if the user row does not exist the DELETE is
    a no-op and no exception is raised.
    """
    # Step 1: Write the audit event BEFORE deletion so the FK is still valid
    #         when the row is inserted.  After the user row is deleted,
    #         ON DELETE SET NULL will null-out audit_log.user_id automatically.
    audit_entry = AuditLog(
        user_id=user_id,
        event_type="user_data_deleted",
        details_json={"reason": "user_request", "initiated_by": "user"},
    )
    session.add(audit_entry)

    # Step 2: Delete the User row.  ON DELETE CASCADE removes all personal-
    #         data rows; ON DELETE SET NULL preserves the audit entry above
    #         (with user_id becoming NULL).
    await session.execute(delete(User).where(User.id == user_id))
