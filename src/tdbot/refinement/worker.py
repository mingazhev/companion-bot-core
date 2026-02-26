"""Refinement worker: dequeue jobs and apply prompt refinements.

One worker instance runs concurrently alongside the Telegram bot.  It blocks
on the Redis refinement queue, processes each job, and handles retry /
dead-letter logic for repeated failures.

Public surface:
    check_and_clear_user_notice(redis, user_id)  — consume pending "updated" notice
    process_one_job(job_data, *, ...)             — process a single queued job
    run_worker(*, ...)                            — main loop (run until cancelled)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import update as sql_update

from tdbot.db.engine import get_async_session
from tdbot.db.models import AuditLog, Job
from tdbot.inference.circuit_breaker import CircuitBreakerOpen
from tdbot.logging_config import get_logger
from tdbot.metrics import REFINEMENT_JOBS
from tdbot.orchestrator.context_loader import load_recent_messages
from tdbot.privacy.field_encryption import NOOP_ENCRYPTOR, FieldEncryptor
from tdbot.prompt.merge_builder import build_system_prompt, extract_base_template, extract_section
from tdbot.prompt.schemas import PromptComponents, SnapshotRecord
from tdbot.redis.queues import (
    QUEUE_REFINEMENT_JOBS,
    QUEUE_RETRY_JOBS,
    dequeue_job,
    enqueue_refinement_job,
    enqueue_retry_job,
)
from tdbot.refinement.client import refine_prompt
from tdbot.refinement.validator import validate_refinement_result

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine

    from tdbot.inference.client import ChatAPIClient
    from tdbot.prompt.snapshot_store import SnapshotStore
    from tdbot.refinement.schemas import SnapshotDelta

log = get_logger(__name__)

# Maximum number of processing attempts before a job is dead-lettered.
MAX_ATTEMPTS = 3

# Maximum number of circuit-breaker re-enqueues before a job is dead-lettered.
# During a sustained provider outage, the worker polls every ~30 seconds,
# so 20 retries ≈ 10 minutes of back-off before giving up.
MAX_CIRCUIT_BREAKER_RETRIES = 20

# Redis key for the per-user "profile updated" notice.
_NOTICE_KEY_PREFIX = "refinement:notice"
_NOTICE_TTL_SECONDS = 86400  # 24 hours


async def _set_user_notice(redis: Redis, user_id: str) -> None:
    """Set a flag in Redis so the bot can surface a 'profile updated' notice."""
    key = f"{_NOTICE_KEY_PREFIX}:{user_id}"
    await redis.set(key, "1", ex=_NOTICE_TTL_SECONDS)


async def check_and_clear_user_notice(redis: Redis, user_id: str) -> bool:
    """Return True (and clear) if a refinement notice is pending for *user_id*.

    Call this after each bot reply to inform the user when their profile has
    been silently refined since their last message.
    """
    key = f"{_NOTICE_KEY_PREFIX}:{user_id}"
    value: str | None = await redis.getdel(key)
    return value is not None


def _apply_delta(snapshot: SnapshotRecord, proposed_delta: SnapshotDelta) -> SnapshotRecord:
    """Return a new ``SnapshotRecord`` with *proposed_delta* applied.

    The existing ``system_prompt`` is used as the base system template so that
    core rules are preserved.  Delta fields that are ``None`` are omitted from
    the new merged prompt (and therefore not layered on top of the base).

    Note: skill_packs from the delta replace the entire existing map when
    provided; otherwise the existing map from ``snapshot.skill_prompts_json``
    is carried forward.
    """
    new_skill_packs: dict[str, str] = (
        proposed_delta.skill_packs
        if proposed_delta.skill_packs is not None
        else {k: str(v) for k, v in snapshot.skill_prompts_json.items()}
    )
    existing_prompt = snapshot.system_prompt
    components = PromptComponents(
        base_system_template=extract_base_template(existing_prompt),
        persona_segment=(
            proposed_delta.persona_segment
            if proposed_delta.persona_segment is not None
            else extract_section(existing_prompt, "Persona")
        ),
        skill_packs=new_skill_packs,
        long_term_profile=(
            proposed_delta.long_term_profile
            if proposed_delta.long_term_profile is not None
            else extract_section(existing_prompt, "Long-term Profile")
        ),
    )
    return SnapshotRecord(
        user_id=snapshot.user_id,
        version=1,  # placeholder; caller overwrites via model_copy before saving
        system_prompt=build_system_prompt(components),
        skill_prompts_json=dict(new_skill_packs),
        source="refinement",
    )


async def process_one_job(
    job_data: dict[str, Any],
    *,
    redis: Redis,
    snapshot_store: SnapshotStore,
    chat_client: ChatAPIClient,
    engine: AsyncEngine,
    encryptor: FieldEncryptor | None = None,
) -> None:
    """Process a single refinement job from the queue.

    Flow:
    1. Parse user_id from *job_data*.
    2. Create a ``Job`` row (status=running).
    3. Load the user's active prompt snapshot.
    4. Load recent conversation messages.
    5. Call ``refine_prompt`` to get a proposed delta.
    6. Validate the delta with ``validate_refinement_result``.
    7. Apply the delta, save the new snapshot, emit audit event.
    8. Update the ``Job`` row to done/failed/dead_letter.

    On failure the job is retried (via the retry queue) up to ``MAX_ATTEMPTS``
    times before being dead-lettered.
    """
    raw_user_id = job_data.get("user_id", "")
    attempt = int(job_data.get("attempt", 0)) + 1

    try:
        user_id = uuid.UUID(str(raw_user_id))
    except ValueError:
        log.error("refinement_invalid_user_id", user_id=raw_user_id)
        return

    user_id_str = str(user_id)
    job_start = time.perf_counter()
    log.info("refinement_job_started", user_id=user_id_str, attempt=attempt)

    # --- Create DB Job row ---
    job_row_id = uuid.uuid4()
    try:
        async with get_async_session(engine) as session:
            session.add(
                Job(
                    id=job_row_id,
                    type="refinement",
                    user_id=user_id,
                    status="running",
                    payload_json=job_data,
                    started_at=datetime.now(tz=UTC),
                    attempt=attempt,
                )
            )
    except Exception as exc:  # noqa: BLE001
        log.error("refinement_job_db_create_failed", user_id=user_id_str, error=str(exc))
        REFINEMENT_JOBS.labels(status="failed").inc()
        return

    # State variables updated throughout the try block.
    final_status = "done"
    error_msg: str | None = None
    job_requeued = False

    try:
        # --- Load active snapshot ---
        snapshot = await snapshot_store.get_active(user_id)
        if snapshot is None:
            log.info("refinement_skipped_no_snapshot", user_id=user_id_str)
            final_status = "skipped"
            return

        # --- Load recent messages ---
        async with get_async_session(engine) as session:
            enc = encryptor or NOOP_ENCRYPTOR
            recent_messages = await load_recent_messages(
                session, user_id, limit=30, encryptor=enc,
            )

        # --- Call refinement model ---
        result = await refine_prompt(chat_client, snapshot, recent_messages)

        # --- Validate output ---
        violations = validate_refinement_result(result)
        if violations:
            log.warning(
                "refinement_policy_violation",
                user_id=user_id_str,
                violations=violations,
            )
            error_msg = "; ".join(violations)
            final_status = "failed"
            return

        # --- Apply delta and store new snapshot ---
        new_snap = _apply_delta(snapshot, result.proposed_delta)

        # Skip save if the delta produced no effective change.
        if (
            new_snap.system_prompt == snapshot.system_prompt
            and new_snap.skill_prompts_json == snapshot.skill_prompts_json
        ):
            log.info("refinement_job_noop", user_id=user_id_str)
            final_status = "skipped"
            return

        new_version = await snapshot_store.next_version(user_id)
        new_snap = new_snap.model_copy(update={"version": new_version})
        await snapshot_store.save(new_snap)
        await snapshot_store.set_active(user_id, new_snap.id)

        # --- Emit audit event ---
        async with get_async_session(engine) as session:
            session.add(
                AuditLog(
                    user_id=user_id,
                    event_type="prompt_refined",
                    details_json={
                        "snapshot_id_before": str(snapshot.id),
                        "snapshot_id_after": str(new_snap.id),
                        "version_before": snapshot.version,
                        "version_after": new_version,
                        "rationale": result.rationale,
                    },
                )
            )

        # --- Notify user (optional, non-blocking) ---
        await _set_user_notice(redis, user_id_str)

        log.info(
            "refinement_job_done",
            user_id=user_id_str,
            version_before=snapshot.version,
            version_after=new_version,
        )

    except CircuitBreakerOpen:
        # The model provider is down — re-enqueue to the primary queue WITHOUT
        # incrementing the attempt counter so this transient condition does not
        # burn through retries.  The worker's 30-second poll timeout on the
        # primary queue provides a natural back-off before the job is retried.
        cb_retries = int(job_data.get("cb_retries", 0)) + 1
        if cb_retries > MAX_CIRCUIT_BREAKER_RETRIES:
            final_status = "dead_letter"
            error_msg = "circuit breaker open — max retries exhausted"
            log.error(
                "refinement_job_dead_lettered_circuit_open",
                user_id=user_id_str,
                cb_retries=cb_retries,
            )
        else:
            cb_payload = {**job_data, "cb_retries": cb_retries}
            await enqueue_refinement_job(redis, user_id_str, cb_payload)
            job_requeued = True
            final_status = "failed"
            error_msg = "circuit breaker open"
            log.warning(
                "refinement_job_deferred_circuit_open",
                user_id=user_id_str,
                attempt=attempt,
                cb_retries=cb_retries,
            )

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        log.error(
            "refinement_job_failed",
            user_id=user_id_str,
            attempt=attempt,
            error=error_msg,
        )
        if attempt >= MAX_ATTEMPTS:
            final_status = "dead_letter"
            log.error(
                "refinement_job_dead_lettered",
                user_id=user_id_str,
                attempts=attempt,
            )
        else:
            # Re-enqueue with incremented attempt counter.
            retry_payload: dict[str, Any] = {**job_data, "attempt": attempt}
            await enqueue_retry_job(redis, user_id_str, retry_payload)
            job_requeued = True
            final_status = "failed"
            log.warning(
                "refinement_job_retried",
                user_id=user_id_str,
                next_attempt=attempt + 1,
            )
    finally:
        try:
            async with get_async_session(engine) as session:
                await session.execute(
                    sql_update(Job)
                    .where(Job.id == job_row_id)
                    .values(
                        status=final_status,
                        finished_at=datetime.now(tz=UTC),
                        error=error_msg,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "refinement_job_db_update_failed",
                user_id=user_id_str,
                error=str(exc),
            )

        # Release the dedup guard so new triggers can enqueue jobs, unless this
        # job was re-enqueued (retry or circuit-breaker) and is still in flight.
        if not job_requeued:
            try:
                await redis.delete(f"refinement:pending:{user_id}")
            except Exception:  # noqa: BLE001
                log.warning("refinement_guard_release_failed", user_id=user_id_str)

        REFINEMENT_JOBS.labels(status=final_status).inc()
        elapsed_ms = round((time.perf_counter() - job_start) * 1000, 2)
        log.info(
            "refinement_job_finished",
            user_id=user_id_str,
            status=final_status,
            attempt=attempt,
            elapsed_ms=elapsed_ms,
        )


async def run_worker(
    *,
    redis: Redis,
    snapshot_store: SnapshotStore,
    chat_client: ChatAPIClient,
    engine: AsyncEngine,
    encryptor: FieldEncryptor | None = None,
    poll_timeout: int = 30,
) -> None:
    """Run the refinement worker loop indefinitely.

    Processes retry-queue jobs first (non-blocking), then blocks on the
    primary refinement queue.  Continues until the task is cancelled.

    Args:
        redis:          Async Redis client.
        snapshot_store: Prompt snapshot store.
        chat_client:    ``ChatAPIClient`` configured for the refinement model.
        engine:         Async SQLAlchemy engine for DB sessions.
        encryptor:      Optional field encryptor for decrypting message content.
        poll_timeout:   Seconds to block waiting for a primary queue job.
    """
    log.info("refinement_worker_started")
    kwargs: dict[str, Any] = {
        "redis": redis,
        "snapshot_store": snapshot_store,
        "chat_client": chat_client,
        "engine": engine,
        "encryptor": encryptor,
    }
    try:
        while True:
            # Non-blocking check on retry queue first.
            retry_job = await dequeue_job(redis, QUEUE_RETRY_JOBS, timeout=1)
            if retry_job is not None:
                await process_one_job(retry_job, **kwargs)
                continue  # drain retry queue before blocking

            # Blocking wait on primary refinement queue.
            job = await dequeue_job(redis, QUEUE_REFINEMENT_JOBS, timeout=poll_timeout)
            if job is None:
                continue  # timeout — loop back

            await process_one_job(job, **kwargs)
    except asyncio.CancelledError:
        log.info("refinement_worker_stopped")
        raise
