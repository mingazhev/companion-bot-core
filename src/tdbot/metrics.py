"""Prometheus metrics registry for TdBot observability.

All metric objects are module-level singletons registered with the default
Prometheus registry on first import.  Expose them via the ``GET /metrics``
endpoint on the internal HTTP service for scraping.

Metric catalogue
----------------
``tdbot_chat_latency_seconds``
    Histogram.  End-to-end latency of the message-processing pipeline
    (ingress receipt → reply sent).  Labels: ``model``.

``tdbot_detector_classifications_total``
    Counter.  Cumulative behavior-change classifications.
    Labels: ``intent``, ``action``, ``risk_level``.

``tdbot_behavior_change_confirmations_total``
    Counter.  Confirmation-dialogue outcomes.
    Labels: ``outcome`` (``confirmed`` | ``cancelled`` | ``superseded``).

``tdbot_behavior_change_reversals_total``
    Counter.  Detector precision proxy — changes cancelled after an initial
    confirmation.  Labels: ``intent``.

``tdbot_refinement_jobs_total``
    Counter.  Cumulative refinement job completions by final status.
    Labels: ``status`` (``done`` | ``failed`` | ``dead_letter``).

``tdbot_prompt_rollbacks_total``
    Counter.  Cumulative prompt snapshot rollbacks.
    Labels: ``reason`` (``manual`` | ``quality_check`` | ``user_command``).

``tdbot_tokens_used_total``
    Counter.  Token consumption by provider, model, and token type.
    Labels: ``provider``, ``model``, ``token_type``
    (``prompt`` | ``completion`` | ``total``).

``tdbot_internal_requests_total``
    Counter.  HTTP requests to ``/internal/*`` endpoints.
    Labels: ``endpoint``, ``status`` (``success`` | ``error``).

``tdbot_internal_request_latency_seconds``
    Histogram.  Latency of ``/internal/*`` HTTP handler execution.
    Labels: ``endpoint``.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# Chat pipeline latency
# ---------------------------------------------------------------------------

CHAT_LATENCY: Histogram = Histogram(
    "tdbot_chat_latency_seconds",
    "End-to-end latency of the message-processing pipeline (ingress → reply).",
    labelnames=["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

# ---------------------------------------------------------------------------
# Behavior change detector
# ---------------------------------------------------------------------------

DETECTOR_CLASSIFICATIONS: Counter = Counter(
    "tdbot_detector_classifications_total",
    "Cumulative behavior-change classifications by intent, action, and risk level.",
    labelnames=["intent", "action", "risk_level"],
)

BEHAVIOR_CHANGE_CONFIRMATIONS: Counter = Counter(
    "tdbot_behavior_change_confirmations_total",
    "Confirmation-dialogue outcomes: confirmed, cancelled, or superseded.",
    labelnames=["outcome"],
)

BEHAVIOR_CHANGE_REVERSALS: Counter = Counter(
    "tdbot_behavior_change_reversals_total",
    "Detector precision proxy: change cancelled after initial confirmation.",
    labelnames=["intent"],
)

# ---------------------------------------------------------------------------
# Refinement worker
# ---------------------------------------------------------------------------

REFINEMENT_JOBS: Counter = Counter(
    "tdbot_refinement_jobs_total",
    "Cumulative refinement job completions by final status.",
    labelnames=["status"],
)

PROMPT_ROLLBACKS: Counter = Counter(
    "tdbot_prompt_rollbacks_total",
    "Cumulative prompt snapshot rollbacks.",
    labelnames=["reason"],
)

# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

TOKENS_USED: Counter = Counter(
    "tdbot_tokens_used_total",
    "Token consumption by provider, model, and token type.",
    labelnames=["provider", "model", "token_type"],
)

# ---------------------------------------------------------------------------
# Internal HTTP service
# ---------------------------------------------------------------------------

INTERNAL_REQUESTS: Counter = Counter(
    "tdbot_internal_requests_total",
    "HTTP requests to /internal/* endpoints.",
    labelnames=["endpoint", "status"],
)

INTERNAL_REQUEST_LATENCY: Histogram = Histogram(
    "tdbot_internal_request_latency_seconds",
    "Latency of /internal/* HTTP handler execution.",
    labelnames=["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)
