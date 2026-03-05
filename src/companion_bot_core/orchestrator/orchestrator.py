"""Conversation orchestrator: full message-processing pipeline.

Flow per incoming user message
-------------------------------
0a. Check Redis for an active abuse block; if blocked, return block message.
0b. Run policy guardrails (prompt injection, unsafe role change, risky
    capability). If any fires, record a violation, potentially trigger an
    abuse block, and return the guardrail's refusal message.
1. Check Redis for a pending medium-risk confirmation.
   - If present and user replies "yes" → record event as confirmed + applied,
     clear pending state, enqueue optional refinement trigger.
   - If present and user replies "no" → record event as not applied, clear.
   - If present with any other text → clear pending state and proceed normally.
2. Classify the message with the behavior change detector.
3. Route by risk / action:
   - ``refuse``      → log + return refusal message (no inference call).
   - ``confirm``     → store pending change, ask confirmation question.
   - ``auto_apply``  → generate reply with current snapshot, then apply
     change to snapshot (takes effect from the next message).
   - ``pass_through``→ generate reply without recording a behavior event.
4. Assemble per-user context (active snapshot + recent history).
5. Call the inference adapter; handle ``CircuitBreakerOpen``.
6. Persist user and assistant :class:`~companion_bot_core.db.models.ConversationMessage` rows.
7. Increment the activity counter and enqueue a refinement job when the
   configured threshold is reached.
8. Return the reply text to the caller.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from companion_bot_core.behavior import classify
from companion_bot_core.behavior.emotion import EMOTION_INSTRUCTIONS, detect_emotion
from companion_bot_core.behavior.extractor import (
    VALID_TONES,
    extract_persona_name,
    extract_skill_topic,
    extract_tone,
)
from companion_bot_core.db.models import BehaviorChangeEvent, ConversationMessage
from companion_bot_core.i18n import normalize_locale, tr
from companion_bot_core.inference import generate_reply, generate_reply_stream
from companion_bot_core.inference.circuit_breaker import CircuitBreakerOpen
from companion_bot_core.logging_config import get_logger
from companion_bot_core.metrics import (
    BEHAVIOR_CHANGE_CONFIRMATIONS,
    BEHAVIOR_CHANGE_REVERSALS,
    CHAT_LATENCY,
    DETECTOR_CLASSIFICATIONS,
    EMOTION_DETECTED,
    FAREWELL_DETECTED,
    GUARDRAIL_BLOCKS,
    REPETITION_GUARD_TRIGGERED,
    RESPONSE_LENGTH_SENTENCES,
    SESSION_MESSAGES,
    TOKENS_USED,
    TOPIC_SWITCH,
)
from companion_bot_core.orchestrator.context_loader import load_user_context
from companion_bot_core.orchestrator.dialogue_state import (
    PendingChange,
    clear_pending_change,
    get_pending_change,
    set_pending_change,
)
from companion_bot_core.orchestrator.response_filter import (
    build_anti_repetition_instruction,
    check_repetition,
)
from companion_bot_core.orchestrator.topic_tracker import (
    TOPIC_SWITCH_INSTRUCTION,
    detect_topic_switch,
    get_stored_keywords,
    store_topic,
)
from companion_bot_core.policy.abuse_throttle import (
    is_user_abuse_blocked,
    record_policy_violation,
)
from companion_bot_core.policy.guardrails import (
    check_prompt_injection,
    check_risky_capability,
    check_unsafe_role_change,
)
from companion_bot_core.privacy.field_encryption import NOOP_ENCRYPTOR, FieldEncryptor
from companion_bot_core.prompt.helpers import get_or_create_profile, rebuild_and_save_snapshot
from companion_bot_core.prompt.merge_builder import (
    build_system_prompt,
    extract_base_template,
    extract_section,
)
from companion_bot_core.prompt.schemas import (
    DEFAULT_SYSTEM_TEMPLATE,
    PromptComponents,
    SnapshotRecord,
)
from companion_bot_core.quality.checks import count_sentences
from companion_bot_core.redis.queues import enqueue_refinement_job
from companion_bot_core.refinement.scheduler import enqueue_if_cadence_due
from companion_bot_core.tracing import span

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from uuid import UUID

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from companion_bot_core.behavior.schemas import DetectedIntent, DetectionResult
    from companion_bot_core.inference.client import ChatAPIClient
    from companion_bot_core.prompt.snapshot_store import SnapshotStore

log = get_logger(__name__)

# Words that confirm a pending medium-risk change (case-insensitive, post-strip)
_CONFIRM_WORDS: frozenset[str] = frozenset(
    {
        "yes", "y", "confirm", "ok", "sure", "yep", "yeah",
        "да", "д", "ага", "угу", "подтверждаю", "подтвердить", "хорошо",
    }
)
# Words that cancel a pending medium-risk change
_CANCEL_WORDS: frozenset[str] = frozenset(
    {
        "no", "n", "cancel", "nope", "nevermind", "never mind",
        "нет", "н", "отмена", "отменить", "неа",
    }
)

# Redis key for the per-user activity counter used to trigger refinement
_ACTIVITY_KEY_PREFIX = "activity_count"
_ACTIVITY_KEY_TTL = 86400  # 1 day

# SET-NX guard key preventing duplicate refinement enqueues per user.
_REFINEMENT_GUARD_PREFIX = "refinement:pending"
_REFINEMENT_GUARD_TTL = 600  # 10 minutes

# Session message counter — auto-expires after 30 min of inactivity.
_SESSION_COUNT_PREFIX = "session:messages"
_SESSION_COUNT_TTL = 1800  # 30 minutes
# Stores the message count of the previous (completed) session.
_SESSION_PREV_COUNT_PREFIX = "session:prev_count"
_SESSION_PREV_COUNT_TTL = 86400  # 1 day

# User-facing messages (used only by tests; production code uses tr())
_CIRCUIT_OPEN_MSG = tr("orchestrator.circuit_open", "en")
_REFUSE_MSG = tr("orchestrator.refuse", "en")
_CHANGE_APPLIED_MSG = tr("orchestrator.change_applied", "en")
_CHANGE_CANCELLED_MSG = tr("orchestrator.change_cancelled", "en")


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


async def _apply_behavior_change(
    intent: DetectedIntent,
    message_text: str,
    user_id: UUID,
    session: AsyncSession,
    snapshot_store: SnapshotStore,
    encryptor: FieldEncryptor | None = None,
) -> bool:
    """Apply a detected behavior change to the user's profile and prompt snapshot.

    Extracts parameters from *message_text* based on *intent*, updates the
    ``UserProfile`` row, and rebuilds the active prompt snapshot.

    Returns ``True`` if the change was successfully applied, ``False`` if the
    required parameter could not be extracted from the message.
    """
    enc = encryptor or NOOP_ENCRYPTOR

    if intent == "tone_change":
        tone = extract_tone(message_text)
        if tone is None or tone not in VALID_TONES:
            log.info(
                "behavior_change_tone_extraction_failed",
                user_id=str(user_id),
                message_text=message_text[:100],
            )
            return False
        profile = await get_or_create_profile(session, user_id)
        profile.tone = enc.encrypt(tone)
        # Decrypt existing persona_name (may be encrypted in DB) for prompt building.
        raw_persona = (
            enc.decrypt_safe(profile.persona_name, default="")
            if profile.persona_name
            else None
        )
        await rebuild_and_save_snapshot(
            snapshot_store, user_id, raw_persona, tone,
            source="behavior_change", session=session,
        )
        log.info(
            "behavior_change_snapshot_updated",
            user_id=str(user_id),
            intent=intent,
            tone=tone,
        )
        return True

    if intent == "persona_change":
        name = extract_persona_name(message_text)
        if name is None:
            log.info(
                "behavior_change_persona_extraction_failed",
                user_id=str(user_id),
                message_text=message_text[:100],
            )
            return False
        profile = await get_or_create_profile(session, user_id)
        profile.persona_name = enc.encrypt(name)
        # Decrypt existing tone (may be encrypted in DB) for prompt building.
        raw_tone = (
            enc.decrypt_safe(profile.tone, default="") if profile.tone else None
        )
        await rebuild_and_save_snapshot(
            snapshot_store, user_id, name, raw_tone,
            source="behavior_change", session=session,
        )
        log.info(
            "behavior_change_snapshot_updated",
            user_id=str(user_id),
            intent=intent,
            persona_name=name,
        )
        return True

    if intent == "skill_add_prompt":
        topic = extract_skill_topic(message_text)
        if topic is None:
            log.info(
                "behavior_change_skill_extraction_failed",
                user_id=str(user_id),
                message_text=message_text[:100],
            )
            return False
        await _add_skill_to_snapshot(snapshot_store, user_id, topic, session=session)
        log.info(
            "behavior_change_snapshot_updated",
            user_id=str(user_id),
            intent=intent,
            skill_topic=topic,
        )
        return True

    if intent == "skill_remove":
        topic = extract_skill_topic(message_text)
        if topic is None:
            log.info(
                "behavior_change_skill_extraction_failed",
                user_id=str(user_id),
                message_text=message_text[:100],
            )
            return False
        removed = await _remove_skill_from_snapshot(
            snapshot_store, user_id, topic, session=session,
        )
        if removed:
            log.info(
                "behavior_change_snapshot_updated",
                user_id=str(user_id),
                intent=intent,
                skill_topic=topic,
            )
        return removed

    return False


async def _add_skill_to_snapshot(
    snapshot_store: SnapshotStore,
    user_id: UUID,
    topic: str,
    *,
    session: AsyncSession | None = None,
) -> None:
    """Add a skill prompt fragment to the user's active snapshot."""
    current = await snapshot_store.get_active(user_id)

    if current is not None:
        base_template = extract_base_template(current.system_prompt)
        skill_packs = {k: str(v) for k, v in current.skill_prompts_json.items()}
        long_term_profile = extract_section(current.system_prompt, "Long-term Profile")
        persona_segment = extract_section(current.system_prompt, "Persona")
    else:
        base_template = DEFAULT_SYSTEM_TEMPLATE
        skill_packs = {}
        long_term_profile = ""
        persona_segment = ""

    skill_key = topic.lower().replace(" ", "_")
    skill_packs[skill_key] = f"Assist the user with {topic}-related questions and tasks."
    raw_skills = dict(skill_packs)

    components = PromptComponents(
        base_system_template=base_template,
        persona_segment=persona_segment,
        skill_packs=skill_packs,
        long_term_profile=long_term_profile,
    )
    system_prompt = build_system_prompt(components)

    version = await snapshot_store.next_version(user_id)
    record = SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt=system_prompt,
        skill_prompts_json=raw_skills,
        source="behavior_change",
    )
    await snapshot_store.save(record, session=session)
    await snapshot_store.set_active(user_id, record.id, session=session)


async def _remove_skill_from_snapshot(
    snapshot_store: SnapshotStore,
    user_id: UUID,
    topic: str,
    *,
    session: AsyncSession | None = None,
) -> bool:
    """Remove a skill prompt fragment from the user's active snapshot.

    Returns ``True`` if a matching skill was found and removed.
    """
    current = await snapshot_store.get_active(user_id)
    if current is None:
        return False

    base_template = extract_base_template(current.system_prompt)
    skill_packs = {k: str(v) for k, v in current.skill_prompts_json.items()}
    long_term_profile = extract_section(current.system_prompt, "Long-term Profile")
    persona_segment = extract_section(current.system_prompt, "Persona")

    skill_key = topic.lower().replace(" ", "_")
    if skill_key not in skill_packs:
        return False
    del skill_packs[skill_key]

    raw_skills = dict(skill_packs)

    components = PromptComponents(
        base_system_template=base_template,
        persona_segment=persona_segment,
        skill_packs=skill_packs,
        long_term_profile=long_term_profile,
    )
    system_prompt = build_system_prompt(components)

    version = await snapshot_store.next_version(user_id)
    record = SnapshotRecord(
        user_id=user_id,
        version=version,
        system_prompt=system_prompt,
        skill_prompts_json=raw_skills,
        source="behavior_change",
    )
    await snapshot_store.save(record, session=session)
    await snapshot_store.set_active(user_id, record.id, session=session)
    return True


async def _persist_messages(
    session: AsyncSession,
    user_id: UUID,
    user_text: str,
    assistant_text: str,
    tokens_used: int,
    model: str,
    ttl_seconds: int,
    encryptor: FieldEncryptor | None = None,
) -> None:
    enc = encryptor or NOOP_ENCRYPTOR
    ttl_expires = datetime.now(tz=UTC) + timedelta(seconds=ttl_seconds)

    session.add(
        ConversationMessage(
            user_id=user_id,
            role="user",
            content=enc.encrypt(user_text),
            model=model,
            ttl_expires_at=ttl_expires,
        )
    )
    session.add(
        ConversationMessage(
            user_id=user_id,
            role="assistant",
            content=enc.encrypt(assistant_text),
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
    """Increment activity counter; enqueue a refinement job at threshold.

    Uses ``>=`` for the threshold check so that a transient Redis failure
    in the ``finally`` cleanup (which resets the counter) cannot
    permanently suppress future triggers.

    A SET-NX guard key (``refinement:pending:{user_id}``) prevents
    duplicate enqueues under concurrent load — only the first request to
    acquire the guard actually pushes a job.  The guard expires after
    10 minutes so a lost worker cannot permanently block future triggers.

    Cleanup (counter reset) is wrapped in a ``try``/``except`` so that a
    transient Redis failure in the ``finally`` block does not propagate
    up and roll back the caller's DB transaction.
    """
    key = f"{_ACTIVITY_KEY_PREFIX}:{user_id}"
    count = await redis.incr(key)
    if count == 1:
        # Set TTL on first creation so the counter resets after inactivity
        await redis.expire(key, _ACTIVITY_KEY_TTL)

    if count >= activity_threshold:
        # Dedup guard: only one in-flight refinement job per user.
        guard_key = f"{_REFINEMENT_GUARD_PREFIX}:{user_id}"
        acquired = await redis.set(guard_key, "1", nx=True, ex=_REFINEMENT_GUARD_TTL)
        if not acquired:
            log.debug(
                "refinement_enqueue_skipped_guard",
                user_id=str(user_id),
                message_count=count,
            )
            return

        payload = {
            "user_id": str(user_id),
            "trigger": "activity_threshold",
            "count": count,
            "created_at": datetime.now(tz=UTC).isoformat(),
        }
        try:
            await enqueue_refinement_job(redis, str(user_id), payload)
            log.info(
                "refinement_job_enqueued",
                user_id=str(user_id),
                message_count=count,
            )
        except Exception:  # noqa: BLE001
            # Enqueue failed — release the guard key so the next threshold
            # crossing can retry, and leave the counter intact so messages
            # are not lost from the trigger perspective.
            try:
                await redis.delete(guard_key)
            except Exception:  # noqa: BLE001
                log.warning(
                    "refinement_guard_cleanup_failed",
                    user_id=str(user_id),
                    key=guard_key,
                )
            log.warning(
                "refinement_enqueue_failed",
                user_id=str(user_id),
                message_count=count,
            )
            return

        # Reset counter only on successful enqueue.
        try:
            await redis.delete(key)
        except Exception:  # noqa: BLE001
            log.warning(
                "activity_counter_cleanup_failed",
                user_id=str(user_id),
                key=key,
            )


async def _track_session_message(
    redis: Redis,
    user_id_str: str,
    *,
    is_farewell: bool = False,
) -> None:
    """Increment the per-user session message counter and observe session length.

    Uses a Redis key with a 30-minute TTL as a lightweight session boundary.
    When the key doesn't exist (INCR returns 1), the previous session ended
    via inactivity — observe the previous session's length from a companion key.
    On farewell, observe the current session's length.
    """
    key = f"{_SESSION_COUNT_PREFIX}:{user_id_str}"
    prev_key = f"{_SESSION_PREV_COUNT_PREFIX}:{user_id_str}"

    try:
        count = await redis.incr(key)
        await redis.expire(key, _SESSION_COUNT_TTL)

        if count == 1:
            # New session — check if there's a previous session count to observe.
            raw = await redis.get(prev_key)
            if raw is not None:
                SESSION_MESSAGES.observe(int(raw))

        # Always keep prev_count updated so it reflects the latest count.
        await redis.set(prev_key, str(count), ex=_SESSION_PREV_COUNT_TTL)

        if is_farewell:
            SESSION_MESSAGES.observe(count)
    except Exception:  # noqa: BLE001
        log.warning("session_message_tracking_failed", user_id=user_id_str)


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
    model: str = "gpt-5-mini",
    conversation_ttl_seconds: int = 604800,
    refinement_activity_threshold: int = 10,
    refinement_cadence_seconds: int = 3600,
    max_tokens: int = 2048,
    encryptor: FieldEncryptor | None = None,
    locale: str | None = None,
    on_stream_chunk: Callable[[str], Awaitable[None]] | None = None,
    context_message_limit: int = 50,
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
        refinement_cadence_seconds:    Minimum seconds between cadence-triggered
                                       refinement jobs for a single user.
        max_tokens:                    Maximum completion tokens.
        encryptor:                     Optional field encryptor for at-rest
                                       encryption of conversation content and
                                       profile fields.  ``None`` uses a
                                       disabled (pass-through) encryptor.

    Returns:
        Reply text to send back to the user.
    """
    user_id_str = str(user_id)
    ui_locale = normalize_locale(locale) if locale is not None else "en"
    pipeline_start = time.perf_counter()

    async with span("orchestrator.process_message", user_id=user_id_str):
        # ------------------------------------------------------------------
        # Step 0a — Check if user is abuse-blocked
        # ------------------------------------------------------------------
        if await is_user_abuse_blocked(redis, user_id_str):
            log.warning("abuse_block_active", user_id=user_id_str)
            CHAT_LATENCY.labels(model=model).observe(
                time.perf_counter() - pipeline_start
            )
            return tr("orchestrator.abuse_block", ui_locale)

        # ------------------------------------------------------------------
        # Step 0b — Run policy guardrails on raw message text
        # ------------------------------------------------------------------
        for guardrail_check in (
            check_prompt_injection,
            check_unsafe_role_change,
            check_risky_capability,
        ):
            result = guardrail_check(message_text)
            if not result.allowed:
                GUARDRAIL_BLOCKS.labels(
                    violation=result.violation or "unknown"
                ).inc()
                await record_policy_violation(redis, user_id_str)
                log.warning(
                    "guardrail_blocked",
                    user_id=user_id_str,
                    violation=result.violation,
                    confidence=result.confidence,
                )
                CHAT_LATENCY.labels(model=model).observe(
                    time.perf_counter() - pipeline_start
                )
                reason_key = {
                    "prompt_injection": "orchestrator.guardrail.prompt_injection",
                    "unsafe_role_change": "orchestrator.guardrail.unsafe_role_change",
                    "risky_capability": "orchestrator.guardrail.risky_capability",
                }.get(result.violation or "", "orchestrator.refuse")
                return tr(reason_key, ui_locale)

        # ------------------------------------------------------------------
        # Step 1 — Check for a pending medium-risk confirmation dialogue
        # ------------------------------------------------------------------
        pending_was_cancelled = False
        pending = await get_pending_change(redis, user_id_str)
        if pending is not None:
            normalized = message_text.strip().lower().rstrip(".,!?;:")
            if normalized in _CONFIRM_WORDS:
                applied = await _apply_behavior_change(
                    intent=pending.detection_result.intent,
                    message_text=pending.original_message,
                    user_id=user_id,
                    session=session,
                    snapshot_store=snapshot_store,
                    encryptor=encryptor,
                )
                await _record_behavior_event(
                    session,
                    user_id,
                    pending.detection_result,
                    applied=applied,
                    confirmed=True,
                )
                await clear_pending_change(redis, user_id_str)
                if applied:
                    await _maybe_enqueue_refinement(redis, user_id, refinement_activity_threshold)
                BEHAVIOR_CHANGE_CONFIRMATIONS.labels(outcome="confirmed").inc()
                log.info(
                    "behavior_change_confirmed",
                    user_id=user_id_str,
                    intent=pending.detection_result.intent,
                    applied=applied,
                )
                CHAT_LATENCY.labels(model=model).observe(
                    time.perf_counter() - pipeline_start
                )
                return (
                    tr("orchestrator.change_applied", ui_locale)
                    if applied
                    else tr("orchestrator.change_apply_failed", ui_locale)
                )

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
                return tr("orchestrator.change_cancelled", ui_locale)

            # Any other text clears the pending state and proceeds normally
            await clear_pending_change(redis, user_id_str)
            pending_was_cancelled = True
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
            return tr("orchestrator.refuse", ui_locale)

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
            # Use intent-specific natural confirmation text
            confirm_key = f"confirm.{detection.intent}"
            try:
                confirm_text = tr(confirm_key, ui_locale)
            except KeyError:
                confirm_text = tr("confirm.generic", ui_locale)
            return confirm_text

        # ------------------------------------------------------------------
        # Step 4 — Build context and generate reply (auto_apply / pass_through)
        # ------------------------------------------------------------------
        # Wrap Steps 4-7 in try/finally so CHAT_LATENCY is always recorded,
        # even when an unhandled exception occurs mid-pipeline.
        try:
            async with span("prompt_manager.load_context", user_id=user_id_str):
                user_context = await load_user_context(
                    session, snapshot_store, user_id, max_tokens,
                    encryptor=encryptor, locale=locale, redis=redis,
                    context_message_limit=context_message_limit,
                )

            # Step 4b — Emotion detection: inject mode-specific instruction.
            # Skip for auto_apply — the message is a behavior-change request,
            # and emotion instructions would conflict with the change intent.
            detected_farewell = False
            if action != "auto_apply":
                emotion = detect_emotion(message_text)
                EMOTION_DETECTED.labels(mode=emotion.mode).inc()
                if emotion.mode == "farewell":
                    FAREWELL_DETECTED.inc()
                    detected_farewell = True
                emotion_instruction = EMOTION_INSTRUCTIONS[emotion.mode]
            else:
                emotion_instruction = ""
            if emotion_instruction:
                user_context = user_context.model_copy(update={
                    "system_prompt": (
                        f"{user_context.system_prompt}\n\n"
                        f"[EmotionMode]\n{emotion_instruction}"
                    ),
                })
                log.info(
                    "emotion_detected",
                    user_id=user_id_str,
                    mode=emotion.mode,
                    confidence=emotion.confidence,
                )

            # Step 4c — Topic tracking: detect switches and inject instruction.
            # Skip for auto_apply — the message is a behavior-change request.
            if action != "auto_apply":
                try:
                    prev_keywords = await get_stored_keywords(redis, user_id_str)
                    topic_result = detect_topic_switch(message_text, prev_keywords)
                    if topic_result.switched:
                        user_context = user_context.model_copy(update={
                            "system_prompt": (
                                f"{user_context.system_prompt}\n\n"
                                f"[TopicSwitch]\n{TOPIC_SWITCH_INSTRUCTION}"
                            ),
                        })
                        TOPIC_SWITCH.inc()
                        log.info(
                            "topic_switch_detected",
                            user_id=user_id_str,
                            signal_score=topic_result.signal_score,
                            keyword_overlap=topic_result.keyword_overlap,
                        )
                    # Always update stored topic keywords.
                    if topic_result.new_keywords:
                        await store_topic(
                            redis, user_id_str, topic_result.new_keywords,
                            save_previous=topic_result.switched,
                        )
                except Exception:  # noqa: BLE001
                    log.warning("topic_tracker_failed", user_id=user_id_str)

            try:
                async with span("model_adapter.generate_reply", user_id=user_id_str):
                    if on_stream_chunk is not None:
                        inference_reply = await generate_reply_stream(
                            chat_client, user_context, message_text,
                            on_chunk=on_stream_chunk,
                        )
                    else:
                        inference_reply = await generate_reply(
                            chat_client, user_context, message_text
                        )
            except CircuitBreakerOpen:
                log.error("circuit_breaker_open_during_chat", user_id=user_id_str)
                return tr("orchestrator.circuit_open", ui_locale)

            reply_text = inference_reply.reply

            # Log when the model ran out of tokens (finish_reason=length) —
            # the reply is truncated mid-sentence.  Consider raising
            # CHAT_MAX_TOKENS if this happens frequently.
            sf = inference_reply.safety_flags
            if sf.finish_reason == "length":
                log.warning(
                    "inference_truncated_by_max_tokens",
                    user_id=user_id_str,
                    completion_tokens=inference_reply.usage.completion_tokens,
                    max_tokens=max_tokens,
                )

            # Handle content-filtered or refused model responses gracefully
            # instead of forwarding empty/truncated text to the user.
            if sf.content_filtered or sf.refusal:
                log.warning(
                    "inference_safety_flag_triggered",
                    user_id=user_id_str,
                    content_filtered=sf.content_filtered,
                    refusal=sf.refusal,
                    finish_reason=sf.finish_reason,
                )
                reply_text = tr("orchestrator.safety_fallback", ui_locale)

            # ------------------------------------------------------------------
            # Step 4d — Repetition guard: strip repeated phrases from response
            # ------------------------------------------------------------------
            recent_assistant = [
                m.content
                for m in user_context.conversation_history
                if m.role == "assistant"
            ][-5:]
            if recent_assistant:
                rep_result = check_repetition(reply_text, recent_assistant)
                if rep_result.repeated_phrases:
                    if len(rep_result.cleaned_text.split()) >= 3:  # noqa: PLR2004
                        # Option B: stripped text is still coherent
                        REPETITION_GUARD_TRIGGERED.labels(action="strip").inc()
                        log.info(
                            "repetition_guard_stripped",
                            user_id=user_id_str,
                            removed=len(rep_result.repeated_phrases),
                        )
                        reply_text = rep_result.cleaned_text
                    elif not sf.content_filtered and not sf.refusal:
                        # Option A: re-call with anti-repetition instruction (max 1)
                        anti_rep = build_anti_repetition_instruction(
                            rep_result.repeated_phrases
                        )
                        patched_ctx = user_context.model_copy(update={
                            "system_prompt": (
                                f"{user_context.system_prompt}\n\n"
                                f"[RepetitionGuard]\n{anti_rep}"
                            ),
                        })
                        try:
                            retry_reply = await generate_reply(
                                chat_client, patched_ctx, message_text,
                            )
                            reply_text = retry_reply.reply
                            REPETITION_GUARD_TRIGGERED.labels(action="recall").inc()
                            log.info(
                                "repetition_guard_recall",
                                user_id=user_id_str,
                            )
                        except CircuitBreakerOpen:
                            REPETITION_GUARD_TRIGGERED.labels(action="recall_failed").inc()
                            log.warning(
                                "repetition_guard_recall_circuit_open",
                                user_id=user_id_str,
                            )

            # Notify user that their previous pending change was auto-cancelled
            # because they sent an unrelated message instead of yes/no.
            if pending_was_cancelled:
                notice = tr("orchestrator.pending_cancelled", ui_locale)
                reply_text = f"{notice}\n\n---\n\n{reply_text}"

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
                try:
                    applied = await _apply_behavior_change(
                        intent=detection.intent,
                        message_text=message_text,
                        user_id=user_id,
                        session=session,
                        snapshot_store=snapshot_store,
                        encryptor=encryptor,
                    )
                except Exception:
                    log.exception(
                        "behavior_change_apply_failed",
                        user_id=user_id_str,
                        intent=detection.intent,
                    )
                    applied = False
                await _record_behavior_event(
                    session,
                    user_id,
                    detection,
                    applied=applied,
                    confirmed=False,
                )
                log.info(
                    "behavior_change_auto_apply_attempted",
                    user_id=user_id_str,
                    intent=detection.intent,
                    applied=applied,
                )

            # Append any post-inference annotations to reply_text BEFORE
            # persisting so that the stored assistant message matches what the
            # user actually receives.

            # Tone/persona changes are applied silently — the user can check
            # their profile via /memory.  No inline notice is appended.

            # Notify user when a skill_remove auto-apply found no matching skill.
            if (
                action == "auto_apply"
                and detection.intent == "skill_remove"
                and not applied
            ):
                reply_text = (
                    f"{reply_text}\n\n"
                    "(I couldn't find a skill matching that topic to remove.)"
                )

            # Surface clarification question when behavior detection had
            # a nonzero but below-threshold confidence score.
            if detection.clarification_question is not None:
                reply_text = f"{reply_text}\n\n{tr('orchestrator.clarification', ui_locale)}"

            # ------------------------------------------------------------------
            # Step 5b — Quality metrics on final reply text
            # ------------------------------------------------------------------
            RESPONSE_LENGTH_SENTENCES.observe(count_sentences(reply_text))
            await _track_session_message(
                redis, user_id_str, is_farewell=detected_farewell,
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
                    encryptor=encryptor,
                )

            # ------------------------------------------------------------------
            # Step 7 — Enqueue optional refinement triggers
            # ------------------------------------------------------------------
            await _maybe_enqueue_refinement(redis, user_id, refinement_activity_threshold)
            await enqueue_if_cadence_due(redis, user_id_str, refinement_cadence_seconds)

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
