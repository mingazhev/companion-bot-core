"""Prometheus metrics registry for Companion Bot Core observability.

All metric objects are module-level singletons registered with the default
Prometheus registry on first import.  Expose them via the ``GET /metrics``
endpoint on the internal HTTP service for scraping.

Metric catalogue
----------------
``companion_bot_core_chat_latency_seconds``
    Histogram.  End-to-end latency of the message-processing pipeline
    (ingress receipt → reply sent).  Labels: ``model``.

``companion_bot_core_detector_classifications_total``
    Counter.  Cumulative behavior-change classifications.
    Labels: ``intent``, ``action``, ``risk_level``.

``companion_bot_core_behavior_change_confirmations_total``
    Counter.  Confirmation-dialogue outcomes.
    Labels: ``outcome`` (``confirmed`` | ``cancelled`` | ``superseded``).

``companion_bot_core_behavior_change_reversals_total``
    Counter.  Detector precision proxy — changes cancelled after an initial
    confirmation.  Labels: ``intent``.

``companion_bot_core_refinement_jobs_total``
    Counter.  Cumulative refinement job completions by final status.
    Labels: ``status`` (``done`` | ``failed`` | ``dead_letter`` | ``skipped``).

``companion_bot_core_prompt_rollbacks_total``
    Counter.  Cumulative prompt snapshot rollbacks.
    Labels: ``reason`` (``manual`` | ``quality_check`` | ``user_command``).

``companion_bot_core_tokens_used_total``
    Counter.  Token consumption by provider, model, and token type.
    Labels: ``provider``, ``model``, ``token_type``
    (``prompt`` | ``completion`` | ``total``).

``companion_bot_core_emotion_detected_total``
    Counter.  Emotion mode classifications by the pre-inference detector.
    Labels: ``mode``.

``companion_bot_core_topic_switch_total``
    Counter.  Conversation topic switches detected by the topic tracker.

``companion_bot_core_response_length_sentences``
    Histogram.  Number of sentences in bot responses.

``companion_bot_core_session_messages_total``
    Histogram.  Number of messages per conversation session.
    Observed when a session ends (farewell or 30-minute inactivity gap).

``companion_bot_core_farewell_detected_total``
    Counter.  Farewell emotion mode detections.

``companion_bot_core_internal_requests_total``
    Counter.  HTTP requests to ``/internal/*`` endpoints.
    Labels: ``endpoint``, ``status`` (``success`` | ``error``).

``companion_bot_core_internal_request_latency_seconds``
    Histogram.  Latency of ``/internal/*`` HTTP handler execution.
    Labels: ``endpoint``.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# Chat pipeline latency
# ---------------------------------------------------------------------------

CHAT_LATENCY: Histogram = Histogram(
    "companion_bot_core_chat_latency_seconds",
    "End-to-end latency of the message-processing pipeline (ingress → reply).",
    labelnames=["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

# ---------------------------------------------------------------------------
# Behavior change detector
# ---------------------------------------------------------------------------

DETECTOR_CLASSIFICATIONS: Counter = Counter(
    "companion_bot_core_detector_classifications_total",
    "Cumulative behavior-change classifications by intent, action, and risk level.",
    labelnames=["intent", "action", "risk_level"],
)

BEHAVIOR_CHANGE_CONFIRMATIONS: Counter = Counter(
    "companion_bot_core_behavior_change_confirmations_total",
    "Confirmation-dialogue outcomes: confirmed, cancelled, or superseded.",
    labelnames=["outcome"],
)

BEHAVIOR_CHANGE_REVERSALS: Counter = Counter(
    "companion_bot_core_behavior_change_reversals_total",
    "Detector precision proxy: change cancelled after initial confirmation.",
    labelnames=["intent"],
)

# ---------------------------------------------------------------------------
# Refinement worker
# ---------------------------------------------------------------------------

REFINEMENT_JOBS: Counter = Counter(
    "companion_bot_core_refinement_jobs_total",
    "Cumulative refinement job completions by final status.",
    labelnames=["status"],
)

PROMPT_ROLLBACKS: Counter = Counter(
    "companion_bot_core_prompt_rollbacks_total",
    "Cumulative prompt snapshot rollbacks.",
    labelnames=["reason"],
)

# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

TOKENS_USED: Counter = Counter(
    "companion_bot_core_tokens_used_total",
    "Token consumption by provider, model, and token type.",
    labelnames=["provider", "model", "token_type"],
)

# ---------------------------------------------------------------------------
# Policy guardrail layer
# ---------------------------------------------------------------------------

GUARDRAIL_BLOCKS: Counter = Counter(
    "companion_bot_core_guardrail_blocks_total",
    "Messages blocked by the policy guardrail layer.",
    labelnames=["violation"],
)

# ---------------------------------------------------------------------------
# Emotion detector
# ---------------------------------------------------------------------------

EMOTION_DETECTED: Counter = Counter(
    "companion_bot_core_emotion_detected_total",
    "Emotion mode classifications by the pre-inference detector.",
    labelnames=["mode"],
)

# ---------------------------------------------------------------------------
# Repetition guard
# ---------------------------------------------------------------------------

REPETITION_GUARD_TRIGGERED: Counter = Counter(
    "companion_bot_core_repetition_guard_triggered_total",
    "Times the post-inference repetition guard stripped or re-called.",
    labelnames=["action"],
)

# ---------------------------------------------------------------------------
# Topic tracker
# ---------------------------------------------------------------------------

TOPIC_SWITCH: Counter = Counter(
    "companion_bot_core_topic_switch_total",
    "Conversation topic switches detected by the topic tracker.",
)

# ---------------------------------------------------------------------------
# Conversation quality metrics
# ---------------------------------------------------------------------------

RESPONSE_LENGTH_SENTENCES: Histogram = Histogram(
    "companion_bot_core_response_length_sentences",
    "Number of sentences in bot responses.",
    buckets=[1, 2, 3, 5, 7, 10, 15],
)

SESSION_MESSAGES: Histogram = Histogram(
    "companion_bot_core_session_messages_total",
    "Number of messages per conversation session.",
    buckets=[1, 3, 5, 7, 10, 15, 20],
)

FAREWELL_DETECTED: Counter = Counter(
    "companion_bot_core_farewell_detected_total",
    "Farewell emotion mode detections.",
)

# ---------------------------------------------------------------------------
# Internal HTTP service
# ---------------------------------------------------------------------------

INTERNAL_REQUESTS: Counter = Counter(
    "companion_bot_core_internal_requests_total",
    "HTTP requests to /internal/* endpoints.",
    labelnames=["endpoint", "status"],
)

INTERNAL_REQUEST_LATENCY: Histogram = Histogram(
    "companion_bot_core_internal_request_latency_seconds",
    "Latency of /internal/* HTTP handler execution.",
    labelnames=["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)
