"""Hard-delete flow for user personal data.

When a user requests data deletion via ``/delete_my_data``, this module
performs a hard delete that:

1. Writes a minimal audit-log entry *before* deletion (so there is a record
   that a deletion was requested and executed).
2. Deletes the :class:`~companion_bot_core.db.models.User` row.
   All personal-data tables (``user_profiles``, ``prompt_snapshots``,
   ``conversation_messages``, ``memory_compactions``,
   ``behavior_change_events``, ``jobs``) have ``ON DELETE CASCADE`` foreign
   keys and are removed automatically.
3. The ``audit_log`` rows referencing this user have ``ON DELETE SET NULL``,
   so they are preserved with ``user_id = NULL`` — the minimal audit trail
   required for compliance.
4. Redis keys scoped to the internal user_id are deleted so that a
   re-registering user does not inherit stale state (e.g. pending change,
   abuse block, activity counter).

The caller is responsible for committing the database transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from companion_bot_core.db.models import AuditLog, User

if TYPE_CHECKING:
    import uuid

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

# Redis key prefixes that are scoped to the internal user UUID.
_REDIS_KEY_PREFIXES = (
    "pending_change",
    "activity_count",
    "refinement:notice",
    "refinement:last_scheduled",
    "refinement:pending",
    "abuse:violations",
    "abuse:block",
    "prompt_cache",
    "prompt:active",
    "prompt:version",
    "profile:write",
    "feedback:session_count",
    "feedback:pending",
    "feedback:last_asked",
    "topic",
    "topic:prev",
    "session:messages",
    "session:prev_count",
    "checkin:last",
    "last_active",
    "suggestion:last",
    "onboarding",
)

# Redis key prefixes scoped to the Telegram user ID (not internal UUID).
_REDIS_TELEGRAM_KEY_PREFIXES = ("rate_limit:user",)


async def hard_delete_user(
    user_id: uuid.UUID,
    session: AsyncSession,
    redis: Redis | None = None,
    telegram_user_id: int | None = None,
) -> None:
    """Remove all personal data for *user_id*, preserving minimal audit trail.

    Args:
        user_id:          Internal UUID of the user to delete.
        session:          An active :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
                          The caller is responsible for committing the transaction.
        redis:            Optional Redis client.  When provided, all Redis keys scoped
                          to *user_id* are deleted so a re-registering user does not
                          inherit stale state.
        telegram_user_id: Optional Telegram user ID.  When provided alongside *redis*,
                          the per-user rate-limit key is also removed.

    The function is idempotent: if the user row does not exist the DELETE is
    a no-op and no exception is raised.
    """
    # Check whether the user row still exists so that repeated calls do not
    # attempt to insert an audit entry with a dangling FK reference.
    result = await session.execute(
        select(User.id).where(User.id == user_id).with_for_update()
    )
    if result.scalar_one_or_none() is None:
        # User already deleted — nothing to do in the DB.  Fall through to
        # the Redis cleanup below so stale keys are still removed.
        pass
    else:
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

    # Step 3: Remove Redis keys scoped to the internal user UUID so that a
    #         user who re-registers does not inherit stale pending changes,
    #         abuse blocks, or activity counters from their previous account.
    if redis is not None:
        user_id_str = str(user_id)
        keys = [f"{prefix}:{user_id_str}" for prefix in _REDIS_KEY_PREFIXES]
        if telegram_user_id is not None:
            keys += [
                f"{prefix}:{telegram_user_id}" for prefix in _REDIS_TELEGRAM_KEY_PREFIXES
            ]
        await redis.delete(*keys)
        # Remove user from the check-in scheduler sorted set.
        await redis.zrem("checkin:schedule", user_id_str)
