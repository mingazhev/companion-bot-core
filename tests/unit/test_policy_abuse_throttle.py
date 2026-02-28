"""Unit tests for companion_bot_core.policy.abuse_throttle.

Uses fakeredis for in-memory Redis without an external dependency.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

from companion_bot_core.policy.abuse_throttle import (
    ABUSE_BLOCK_MESSAGE,
    clear_abuse_block,
    get_violation_count,
    is_user_abuse_blocked,
    record_policy_violation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def redis() -> FakeRedis:
    client: FakeRedis = FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


# ---------------------------------------------------------------------------
# is_user_abuse_blocked — initial state
# ---------------------------------------------------------------------------


class TestIsUserAbuseBlockedInitialState:
    @pytest.mark.asyncio
    async def test_new_user_is_not_blocked(self, redis: FakeRedis) -> None:
        assert await is_user_abuse_blocked(redis, "user_1") is False

    @pytest.mark.asyncio
    async def test_returns_bool(self, redis: FakeRedis) -> None:
        result = await is_user_abuse_blocked(redis, "user_2")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# record_policy_violation — below threshold
# ---------------------------------------------------------------------------


class TestRecordPolicyViolationBelowThreshold:
    @pytest.mark.asyncio
    async def test_single_violation_does_not_block(self, redis: FakeRedis) -> None:
        blocked = await record_policy_violation(redis, "user_10", block_threshold=5)
        assert blocked is False

    @pytest.mark.asyncio
    async def test_four_violations_do_not_block_at_threshold_5(self, redis: FakeRedis) -> None:
        for _ in range(4):
            blocked = await record_policy_violation(redis, "user_11", block_threshold=5)
        assert blocked is False

    @pytest.mark.asyncio
    async def test_violation_count_increments(self, redis: FakeRedis) -> None:
        for _ in range(3):
            await record_policy_violation(redis, "user_12", block_threshold=10)
        count = await get_violation_count(redis, "user_12")
        assert count == 3

    @pytest.mark.asyncio
    async def test_user_not_blocked_below_threshold(self, redis: FakeRedis) -> None:
        await record_policy_violation(redis, "user_13", block_threshold=5)
        assert await is_user_abuse_blocked(redis, "user_13") is False


# ---------------------------------------------------------------------------
# record_policy_violation — at/above threshold
# ---------------------------------------------------------------------------


class TestRecordPolicyViolationAtThreshold:
    @pytest.mark.asyncio
    async def test_fifth_violation_triggers_block(self, redis: FakeRedis) -> None:
        blocked = False
        for _ in range(5):
            blocked = await record_policy_violation(redis, "user_20", block_threshold=5)
        assert blocked is True

    @pytest.mark.asyncio
    async def test_user_is_blocked_after_threshold(self, redis: FakeRedis) -> None:
        for _ in range(5):
            await record_policy_violation(redis, "user_21", block_threshold=5)
        assert await is_user_abuse_blocked(redis, "user_21") is True

    @pytest.mark.asyncio
    async def test_record_returns_true_on_block(self, redis: FakeRedis) -> None:
        results = [
            await record_policy_violation(redis, "user_22", block_threshold=3)
            for _ in range(3)
        ]
        assert results[-1] is True

    @pytest.mark.asyncio
    async def test_block_with_threshold_1(self, redis: FakeRedis) -> None:
        blocked = await record_policy_violation(redis, "user_23", block_threshold=1)
        assert blocked is True
        assert await is_user_abuse_blocked(redis, "user_23") is True


# ---------------------------------------------------------------------------
# get_violation_count
# ---------------------------------------------------------------------------


class TestGetViolationCount:
    @pytest.mark.asyncio
    async def test_zero_for_new_user(self, redis: FakeRedis) -> None:
        count = await get_violation_count(redis, "user_30")
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_matches_recorded_violations(self, redis: FakeRedis) -> None:
        for _ in range(4):
            await record_policy_violation(redis, "user_31", block_threshold=10)
        assert await get_violation_count(redis, "user_31") == 4

    @pytest.mark.asyncio
    async def test_does_not_add_entry(self, redis: FakeRedis) -> None:
        await record_policy_violation(redis, "user_32", block_threshold=10)
        # Calling get_violation_count should not add an extra entry.
        await get_violation_count(redis, "user_32")
        await get_violation_count(redis, "user_32")
        assert await get_violation_count(redis, "user_32") == 1


# ---------------------------------------------------------------------------
# clear_abuse_block
# ---------------------------------------------------------------------------


class TestClearAbuseBlock:
    @pytest.mark.asyncio
    async def test_clears_active_block(self, redis: FakeRedis) -> None:
        for _ in range(5):
            await record_policy_violation(redis, "user_40", block_threshold=5)
        assert await is_user_abuse_blocked(redis, "user_40") is True

        await clear_abuse_block(redis, "user_40")
        assert await is_user_abuse_blocked(redis, "user_40") is False

    @pytest.mark.asyncio
    async def test_clear_on_non_blocked_user_is_noop(self, redis: FakeRedis) -> None:
        # Should not raise.
        await clear_abuse_block(redis, "user_41")
        assert await is_user_abuse_blocked(redis, "user_41") is False


# ---------------------------------------------------------------------------
# User isolation
# ---------------------------------------------------------------------------


class TestUserIsolation:
    @pytest.mark.asyncio
    async def test_violations_are_per_user(self, redis: FakeRedis) -> None:
        for _ in range(5):
            await record_policy_violation(redis, "user_50", block_threshold=5)

        # user_51 should be unaffected.
        assert await is_user_abuse_blocked(redis, "user_51") is False
        assert await get_violation_count(redis, "user_51") == 0

    @pytest.mark.asyncio
    async def test_different_users_have_independent_counts(self, redis: FakeRedis) -> None:
        await record_policy_violation(redis, "user_52", block_threshold=10)
        await record_policy_violation(redis, "user_52", block_threshold=10)
        await record_policy_violation(redis, "user_53", block_threshold=10)

        assert await get_violation_count(redis, "user_52") == 2
        assert await get_violation_count(redis, "user_53") == 1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_abuse_block_message_is_non_empty_string(self) -> None:
        assert isinstance(ABUSE_BLOCK_MESSAGE, str)
        assert len(ABUSE_BLOCK_MESSAGE) > 0
