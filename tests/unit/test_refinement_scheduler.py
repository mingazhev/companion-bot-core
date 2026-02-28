"""Unit tests for companion_bot_core.refinement.scheduler."""

from __future__ import annotations

import time
from unittest.mock import patch

import fakeredis.aioredis as fakeredis
import pytest

from companion_bot_core.redis.queues import QUEUE_REFINEMENT_JOBS, get_queue_length
from companion_bot_core.refinement.scheduler import (
    _last_scheduled_key,
    enqueue_if_cadence_due,
    record_refinement_scheduled,
    should_schedule_by_cadence,
)

# ---------------------------------------------------------------------------
# should_schedule_by_cadence
# ---------------------------------------------------------------------------


async def test_should_schedule_returns_true_when_no_record() -> None:
    redis = fakeredis.FakeRedis()
    result = await should_schedule_by_cadence(redis, "user-1", cadence_seconds=3600)
    assert result is True


async def test_should_schedule_returns_false_when_recently_scheduled() -> None:
    redis = fakeredis.FakeRedis()
    # Record a timestamp just now
    await redis.set(_last_scheduled_key("user-1"), str(time.time()), ex=7200)
    result = await should_schedule_by_cadence(redis, "user-1", cadence_seconds=3600)
    assert result is False


async def test_should_schedule_returns_true_when_cadence_elapsed() -> None:
    redis = fakeredis.FakeRedis()
    # Record a timestamp far in the past (2 hours ago)
    past = time.time() - 7201
    await redis.set(_last_scheduled_key("user-2"), str(past), ex=7200)
    result = await should_schedule_by_cadence(redis, "user-2", cadence_seconds=3600)
    assert result is True


async def test_should_schedule_exactly_at_cadence_boundary() -> None:
    redis = fakeredis.FakeRedis()
    # Record a timestamp exactly cadence_seconds ago
    cadence = 3600
    past = time.time() - cadence
    await redis.set(_last_scheduled_key("user-3"), str(past), ex=7200)
    result = await should_schedule_by_cadence(redis, "user-3", cadence_seconds=cadence)
    # elapsed >= cadence so should be True
    assert result is True


# ---------------------------------------------------------------------------
# record_refinement_scheduled
# ---------------------------------------------------------------------------


async def test_record_refinement_scheduled_sets_key() -> None:
    redis = fakeredis.FakeRedis()
    cadence = 3600
    with patch("companion_bot_core.refinement.scheduler.time") as mock_time:
        now = 1_700_000_000.0
        mock_time.time.return_value = now
        await record_refinement_scheduled(redis, "user-4", cadence)

    raw = await redis.get(_last_scheduled_key("user-4"))
    assert raw is not None
    assert float(raw) == pytest.approx(1_700_000_000.0, abs=1.0)


async def test_record_refinement_scheduled_sets_ttl() -> None:
    redis = fakeredis.FakeRedis()
    cadence = 3600
    await record_refinement_scheduled(redis, "user-5", cadence)
    ttl = await redis.ttl(_last_scheduled_key("user-5"))
    # TTL should be approximately 2 * cadence
    assert ttl > cadence
    assert ttl <= cadence * 2 + 5  # small tolerance


# ---------------------------------------------------------------------------
# enqueue_if_cadence_due
# ---------------------------------------------------------------------------


async def test_enqueue_if_cadence_due_enqueues_when_no_record() -> None:
    redis = fakeredis.FakeRedis()
    result = await enqueue_if_cadence_due(redis, "user-6", cadence_seconds=3600)
    assert result is True
    queue_len = await get_queue_length(redis, QUEUE_REFINEMENT_JOBS)
    assert queue_len == 1


async def test_enqueue_if_cadence_due_skips_when_recently_scheduled() -> None:
    redis = fakeredis.FakeRedis()
    # First call enqueues
    await enqueue_if_cadence_due(redis, "user-7", cadence_seconds=3600)
    # Second call should be skipped (cadence not elapsed)
    result = await enqueue_if_cadence_due(redis, "user-7", cadence_seconds=3600)
    assert result is False
    # Queue should still have only one job
    queue_len = await get_queue_length(redis, QUEUE_REFINEMENT_JOBS)
    assert queue_len == 1


async def test_enqueue_if_cadence_due_enqueues_after_cadence() -> None:
    redis = fakeredis.FakeRedis()
    cadence = 3600
    # Simulate last scheduled 2 hours ago
    past = time.time() - 7201
    await redis.set(_last_scheduled_key("user-8"), str(past), ex=7200)
    result = await enqueue_if_cadence_due(redis, "user-8", cadence_seconds=cadence)
    assert result is True
    queue_len = await get_queue_length(redis, QUEUE_REFINEMENT_JOBS)
    assert queue_len == 1


async def test_enqueue_if_cadence_due_records_timestamp_after_enqueue() -> None:
    redis = fakeredis.FakeRedis()
    await enqueue_if_cadence_due(redis, "user-9", cadence_seconds=3600)
    raw = await redis.get(_last_scheduled_key("user-9"))
    assert raw is not None
    # Timestamp should be close to now
    assert abs(time.time() - float(raw)) < 5


