"""Unit tests for the Prometheus metrics registry (companion_bot_core.metrics).

Tests verify that:
- All expected metric objects are importable.
- Each metric accepts the declared label sets without raising.
- Counter increments and histogram observations do not raise.

Note: prometheus_client uses a global default registry, so metric
objects persist across tests in the same process.  These tests only
verify that the public interface works correctly; they do not assert
specific counter values.
"""

from __future__ import annotations

import pytest

from companion_bot_core.metrics import (
    BEHAVIOR_CHANGE_CONFIRMATIONS,
    BEHAVIOR_CHANGE_REVERSALS,
    CHAT_LATENCY,
    DETECTOR_CLASSIFICATIONS,
    INTERNAL_REQUEST_LATENCY,
    INTERNAL_REQUESTS,
    PROMPT_ROLLBACKS,
    REFINEMENT_JOBS,
    TOKENS_USED,
)


class TestChatLatency:
    def test_observe_does_not_raise(self) -> None:
        CHAT_LATENCY.labels(model="gpt-4o-mini").observe(1.23)

    def test_multiple_observations(self) -> None:
        hist = CHAT_LATENCY.labels(model="gpt-4o")
        for val in [0.1, 0.5, 2.0, 10.0]:
            hist.observe(val)


class TestDetectorClassifications:
    def test_increment_normal_chat(self) -> None:
        DETECTOR_CLASSIFICATIONS.labels(
            intent="normal_chat", action="pass_through", risk_level="low"
        ).inc()

    def test_increment_high_risk(self) -> None:
        DETECTOR_CLASSIFICATIONS.labels(
            intent="safety_override_attempt", action="refuse", risk_level="high"
        ).inc()

    def test_increment_medium_risk(self) -> None:
        DETECTOR_CLASSIFICATIONS.labels(
            intent="persona_change", action="confirm", risk_level="medium"
        ).inc()


class TestBehaviorChangeConfirmations:
    @pytest.mark.parametrize("outcome", ["confirmed", "cancelled", "superseded"])
    def test_valid_outcomes(self, outcome: str) -> None:
        BEHAVIOR_CHANGE_CONFIRMATIONS.labels(outcome=outcome).inc()


class TestBehaviorChangeReversals:
    def test_increment_with_intent(self) -> None:
        BEHAVIOR_CHANGE_REVERSALS.labels(intent="persona_change").inc()

    def test_increment_tone_change(self) -> None:
        BEHAVIOR_CHANGE_REVERSALS.labels(intent="tone_change").inc()


class TestRefinementJobs:
    @pytest.mark.parametrize("status", ["done", "failed", "dead_letter"])
    def test_valid_statuses(self, status: str) -> None:
        REFINEMENT_JOBS.labels(status=status).inc()


class TestPromptRollbacks:
    @pytest.mark.parametrize("reason", ["manual", "quality_check", "user_command"])
    def test_valid_reasons(self, reason: str) -> None:
        PROMPT_ROLLBACKS.labels(reason=reason).inc()


class TestTokensUsed:
    def test_prompt_tokens(self) -> None:
        TOKENS_USED.labels(
            provider="openai", model="gpt-4o-mini", token_type="prompt"
        ).inc(150)

    def test_completion_tokens(self) -> None:
        TOKENS_USED.labels(
            provider="openai", model="gpt-4o-mini", token_type="completion"
        ).inc(75)

    def test_total_tokens(self) -> None:
        TOKENS_USED.labels(
            provider="openai", model="gpt-4o-mini", token_type="total"
        ).inc(225)

    def test_different_models(self) -> None:
        for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6"]:
            TOKENS_USED.labels(
                provider="openai", model=model, token_type="total"
            ).inc(10)


class TestInternalRequestMetrics:
    def test_requests_counter_success(self) -> None:
        INTERNAL_REQUESTS.labels(endpoint="refine", status="success").inc()

    def test_requests_counter_error(self) -> None:
        INTERNAL_REQUESTS.labels(endpoint="detect_change", status="error").inc()

    def test_latency_histogram(self) -> None:
        INTERNAL_REQUEST_LATENCY.labels(endpoint="refine").observe(0.042)

    def test_latency_histogram_detect(self) -> None:
        INTERNAL_REQUEST_LATENCY.labels(endpoint="detect_change").observe(0.005)
