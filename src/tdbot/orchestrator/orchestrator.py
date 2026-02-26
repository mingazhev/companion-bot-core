"""Conversation orchestrator: full message-processing pipeline.

Flow per incoming user message
-------------------------------
1. Check Redis for a pending medium-risk confirmation.
   - If present and user replies "yes" → record event as confirmed + applied,
     clear pending state, enqueue optional refinement trigger.
   - If present and user replies "no" → record event as not applied, clear.
   - If present with any other text → clear pending state and proceed normally.
2. Classify the message with the behavior change detector.
3. Route by risk / action:
   - ``refuse``      → log + return refusal message (no inference call).
   - ``confirm``     → store pending change, ask confirmation question.
   - ``auto_apply``  → log event as applied, then generate reply.
   - ``pass_through``→ generate reply without recording a behavior event.
4. Assemble per-user context (active snapshot + recent history).
5. Call the inference adapter; handle ``CircuitBreakerOpen``.
6. Persist user and assistant :class:`~tdbot.db.models.ConversationMessage` rows.
7. Increment the activity counter and enqueue a refinement job when the
   configured threshold is reached.
8. Return the reply text to the caller.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from tdbot.behavior import classify
from tdbot.db.models import BehaviorChangeEvent, ConversationMessage
from tdbot.inference import generate_reply
from tdbot.inference.circuit_breaker import CircuitBreakerOpen
from tdbot.logging_config import get_logger
from tdbot.metrics import (
    BEHAVIOR_CHANGE_CONFIRMATIONS,
    BEHAVIOR_CHANGE_REVERSALS,
    CHAT_LATENCY,
    DETECTOR_CLASSIFICATIONS,
    TOKENS_USED,
)
from tdbot.orchestrator.context_loader import load_user_context
from tdbot.orchestrator.dialogue_state import (
    PendingChange,
    clear_pending_change,
    get_pending_change,
    set_pending_change,
)
from tdbot.redis.queues import enqueue_refinement_job
from tdbot.tracing import span

if TYPE_CHECKING:
    from uuid import UUID

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from tdbot.behavior.schemas import DetectionResult
    from tdbot.inference.client import ChatAPIClient
    from tdbot.prompt.snapshot_store import SnapshotStore

log = get_logger(__name__)

# Words that confirm a pending medium-risk change (case-insensitive, post-strip)
_CONFIRM_WORDS: frozenset[str] = frozenset({"yes", "y", "confirm", "ok", "sure", "yep", "yeah"})
# Words that cancel a pending medium-risk change
_CANCEL_WORDS: frozenset[str] = frozenset({"no", "n", "cancel", "nope", "nevermind", "never mind"})

# Redis key for the per-user activity counter used to trigger refinement
_ACTIVITY_KEY_PREFIX = "activity_count"
_ACTIVITY_KEY_TTL = 86400  # 1 day

# User-facing messages
_CIRCUIT_OPEN_MSG = (
    "I'm having trouble reaching the AI service right now. Please try again in a moment."
)
_REFUSE_MSG = (
    "I can't make that change — it conflicts with safety guidelines. "
    "If you'd like to adjust your experience, try /set_tone or /set_persona."
)
_CHANGE_APPLIED_MSG = "Done! I've recorded your preference and will adapt accordingly."
_CHANGE_CANCELLED_MSG = "No problem, keeping things as they are."

_CONFIRM_TEMPLATE = (
    "You'd like to change {intent}. This is a moderate setting change. "
    "Reply 'yes' to confirm or 'no' to cancel."
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _record_behavior_event(
    session: AsyncSession,
    user_id: UUID,
    detection: DetectionResult,
    *,
    applied: bool,
    confirmed: bool,
) -> None:
    event = BehaviorChangeEvent(
        user_id=user_id,
        intent=detection.intent,
        risk_level=detection.risk_level,
        confidence=detection.confidence,
        applied=applied,
        confirmed=confirmed,
    )
    session.add(event)
    await session.flush()


async def _persist_messages(
    session: AsyncSession,
    user_id: UUID,
    user_text: str,
    assistant_text: str,
    tokens_used: int,
    model: str,
    ttl_seconds: int,
) -> None:
    ttl_expires = datetime.now(tz=UTC) + timedelta(seconds=ttl_seconds)

    session.add(
        ConversationMessage(
            user_id=user_id,
            role="user",
            content=user_text,
            model=model,
            ttl_expires_at=ttl_expires,
        )
    )
    session.add(
        ConversationMessage(
            user_id=user_id,
            role="assistant",
            content=assistant_text,
            tokens_used=tokens_used,
            model=model,
            ttl_expires_at=ttl_expires,
        )
    )
    await session.flush()


async def _maybe_enqueue_refinement(
    redis: Redis,
    user_id: UUID,
    activity_threshold: int,
) -> None:
    """Increment activity counter; enqueue a refinement job at threshold."""
    key = f"{_ACTIVITY_KEY_PREFIX}:{user_id}"
    count = await redis.incr(key)
    if count == 1:
        # Set TTL on first creation so the counter resets after inactivity
        await redis.expire(key, _ACTIVITY_KEY_TTL)

    if count >= activity_threshold:
        payload = {
            "user_id": str(user_id),
            "trigger": "activity_threshold",
            "count": count,
        }
        await enqueue_refinement_job(redis, str(user_id), payload)
        await redis.delete(key)
        log.info(
            "refinement_job_enqueued",
            user_id=str(user_id),
            message_count=count,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def process_message(
    user_id: UUID,
    message_text: str,
    session: AsyncSession,
    snapshot_store: SnapshotStore,
    redis: Redis,
    chat_client: ChatAPIClient,
    model: str = "gpt-4o-mini",
    conversation_ttl_seconds: int = 604800,
    refinement_activity_threshold: int = 10,
    max_tokens: int = 1024,
) -> str:
    """Orchestrate a single user message through the full processing pipeline.

    Args:
        user_id:                       Internal UUID of the user.
        message_text:                  Raw text of the user's message.
        session:                       Active async database session.
        snapshot_store:                Prompt snapshot store for the user's persona.
        redis:                         Async Redis client.
        chat_client:                   Configured inference API client.
        model:                         Model identifier for persistence metadata.
        conversation_ttl_seconds:      TTL for conversation message rows.
        refinement_activity_threshold: Messages before enqueueing a refinement job.
        max_tokens:                    Maximum completion tokens.

    Returns:
        Reply text to send back to the user.
    """
    user_id_str = str(user_id)
    pipeline_start = time.perf_counter()

    async with span("orchestrator.process_message", user_id=user_id_str):
        # ------------------------------------------------------------------
        # Step 1 — Check for a pending medium-risk confirmation dialogue
        # ------------------------------------------------------------------
        pending = await get_pending_change(redis, user_id_str)
        if pending is not None:
            normalized = message_text.strip().lower().rstrip(".,!?;:")
            if normalized in _CONFIRM_WORDS:
                await _record_behavior_event(
                    session,
                    user_id,
                    pending.detection_result,
                    applied=True,
                    confirmed=True,
                )
                await clear_pending_change(redis, user_id_str)
                await _maybe_enqueue_refinement(redis, user_id, refinement_activity_threshold)
                BEHAVIOR_CHANGE_CONFIRMATIONS.labels(outcome="confirmed").inc()
                log.info(
                    "behavior_change_confirmed",
                    user_id=user_id_str,
                    intent=pending.detection_result.intent,
                )
                CHAT_LATENCY.labels(model=model).observe(
                    time.perf_counter() - pipeline_start
                )
                return _CHANGE_APPLIED_MSG

            if normalized in _CANCEL_WORDS:
                await _record_behavior_event(
                    session,
                    user_id,
                    pending.detection_result,
                    applied=False,
                    confirmed=False,
                )
                await clear_pending_change(redis, user_id_str)
                BEHAVIOR_CHANGE_CONFIRMATIONS.labels(outcome="cancelled").inc()
                BEHAVIOR_CHANGE_REVERSALS.labels(
                    intent=pending.detection_result.intent
                ).inc()
                log.info(
                    "behavior_change_cancelled",
                    user_id=user_id_str,
                    intent=pending.detection_result.intent,
                )
                CHAT_LATENCY.labels(model=model).observe(
                    time.perf_counter() - pipeline_start
                )
                return _CHANGE_CANCELLED_MSG

            # Any other text clears the pending state and proceeds normally
            await clear_pending_change(redis, user_id_str)
            BEHAVIOR_CHANGE_CONFIRMATIONS.labels(outcome="superseded").inc()
            log.info(
                "behavior_change_pending_cleared_by_unrelated_message",
                user_id=user_id_str,
            )

        # ------------------------------------------------------------------
        # Step 2 — Classify incoming message intent
        # ------------------------------------------------------------------
        async with span("detector.classify", user_id=user_id_str):
            detection = classify(message_text)

        action = detection.action
        DETECTOR_CLASSIFICATIONS.labels(
            intent=detection.intent,
            action=detection.action,
            risk_level=detection.risk_level,
        ).inc()

        # ------------------------------------------------------------------
        # Step 3 — Route by action
        # ------------------------------------------------------------------
        if action == "refuse":
            await _record_behavior_event(
                session,
                user_id,
                detection,
                applied=False,
                confirmed=False,
            )
            log.warning(
                "behavior_change_refused",
                user_id=user_id_str,
                intent=detection.intent,
                risk_level=detection.risk_level,
                confidence=detection.confidence,
            )
            CHAT_LATENCY.labels(model=model).observe(
                time.perf_counter() - pipeline_start
            )
            return _REFUSE_MSG

        if action == "confirm":
            pending_change = PendingChange(
                detection_result=detection,
                original_message=message_text,
            )
            await set_pending_change(redis, user_id_str, pending_change)
            log.info(
                "behavior_change_pending_confirmation",
                user_id=user_id_str,
                intent=detection.intent,
            )
            CHAT_LATENCY.labels(model=model).observe(
                time.perf_counter() - pipeline_start
            )
            return _CONFIRM_TEMPLATE.format(intent=detection.intent.replace("_", " "))

        # ------------------------------------------------------------------
        # Step 4 — Build context and generate reply (auto_apply / pass_through)
        # ------------------------------------------------------------------
        # Wrap Steps 4-7 in try/finally so CHAT_LATENCY is always recorded,
        # even when an unhandled exception occurs mid-pipeline.
        try:
            async with span("prompt_manager.load_context", user_id=user_id_str):
                user_context = await load_user_context(
                    session, snapshot_store, user_id, max_tokens
                )

            try:
                async with span("model_adapter.generate_reply", user_id=user_id_str):
                    inference_reply = await generate_reply(
                        chat_client, user_context, message_text
                    )
            except CircuitBreakerOpen:
                log.error("circuit_breaker_open_during_chat", user_id=user_id_str)
                return _CIRCUIT_OPEN_MSG

            reply_text = inference_reply.reply

            # Record token usage metrics (noqa: S106 — not a password, metric label)
            TOKENS_USED.labels(
                provider="openai", model=model, token_type="prompt"  # noqa: S106
            ).inc(inference_reply.usage.prompt_tokens)
            TOKENS_USED.labels(
                provider="openai", model=model, token_type="completion"  # noqa: S106
            ).inc(inference_reply.usage.completion_tokens)
            TOKENS_USED.labels(
                provider="openai", model=model, token_type="total"  # noqa: S106
            ).inc(inference_reply.usage.total_tokens)

            # ------------------------------------------------------------------
            # Step 5 — Record behavior event for auto-applied changes
            # ------------------------------------------------------------------
            if action == "auto_apply":
                await _record_behavior_event(
                    session,
                    user_id,
                    detection,
                    applied=True,
                    confirmed=False,
                )
                log.info(
                    "behavior_change_auto_applied",
                    user_id=user_id_str,
                    intent=detection.intent,
                )

            # ------------------------------------------------------------------
            # Step 6 — Persist conversation messages
            # ------------------------------------------------------------------
            async with span("persistence.save_messages", user_id=user_id_str):
                await _persist_messages(
                    session=session,
                    user_id=user_id,
                    user_text=message_text,
                    assistant_text=reply_text,
                    tokens_used=inference_reply.usage.total_tokens,
                    model=model,
                    ttl_seconds=conversation_ttl_seconds,
                )

            # ------------------------------------------------------------------
            # Step 7 — Enqueue optional refinement trigger
            # ------------------------------------------------------------------
            await _maybe_enqueue_refinement(redis, user_id, refinement_activity_threshold)

            log.info(
                "chat_pipeline_completed",
                user_id=user_id_str,
                model=model,
                elapsed_ms=round(
                    (time.perf_counter() - pipeline_start) * 1000, 2
                ),
                prompt_tokens=inference_reply.usage.prompt_tokens,
                completion_tokens=inference_reply.usage.completion_tokens,
            )

            return reply_text
        finally:
            CHAT_LATENCY.labels(model=model).observe(
                time.perf_counter() - pipeline_start
            )
