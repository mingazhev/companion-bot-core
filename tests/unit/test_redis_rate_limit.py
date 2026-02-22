"""Unit tests for Redis sliding-window rate limiting."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import fakeredis
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from tdbot.redis.rate_limit import (
    _KEY_GLOBAL,
    _PREFIX_USER,
    check_global_rate_limit,
    check_user_rate_limit,
    get_user_request_count,
)

FakeRedis = fakeredis.FakeAsyncRedis


@pytest_asyncio.fixture
async def redis() -> AsyncGenerator[FakeRedis, None]:
    client = FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


class TestUserRateLimit:
    async def test_first_request_is_allowed(self, redis: FakeRedis) -> None:
        result = await check_user_rate_limit(redis, user_id="u1", max_requests=5)
        assert result is True

    async def test_requests_within_limit_are_allowed(self, redis: FakeRedis) -> None:
        for _ in range(5):
            ok = await check_user_rate_limit(redis, user_id="u1", max_requests=5)
            assert ok is True

    async def test_request_exceeding_limit_is_rejected(self, redis: FakeRedis) -> None:
        # Fill up the limit.
        for _ in range(5):
            await check_user_rate_limit(redis, user_id="u1", max_requests=5)
        # The 6th request must be rejected.
        result = await check_user_rate_limit(redis, user_id="u1", max_requests=5)
        assert result is False

    async def test_different_users_are_isolated(self, redis: FakeRedis) -> None:
        # Fill user A to the limit.
        for _ in range(3):
            await check_user_rate_limit(redis, user_id="alice", max_requests=3)
        # User A is now at limit.
        assert await check_user_rate_limit(redis, user_id="alice", max_requests=3) is False
        # User B is unaffected.
        assert await check_user_rate_limit(redis, user_id="bob", max_requests=3) is True

    async def test_key_uses_user_prefix(self, redis: FakeRedis) -> None:
        await check_user_rate_limit(redis, user_id="u1", max_requests=5)
        keys = await redis.keys(f"{_PREFIX_USER}*")
        assert f"{_PREFIX_USER}u1" in keys

    async def test_key_has_ttl_after_request(self, redis: FakeRedis) -> None:
        await check_user_rate_limit(redis, user_id="u1", max_requests=5, window_seconds=60)
        ttl = await redis.ttl(f"{_PREFIX_USER}u1")
        assert ttl > 0

    async def test_expired_entries_are_not_counted(self, redis: FakeRedis) -> None:
        # Record requests in a tiny 1-second window.
        for _ in range(3):
            await check_user_rate_limit(redis, user_id="u1", max_requests=3, window_seconds=1)
        # Wait for the window to expire.
        await asyncio.sleep(1.1)
        # The window has passed; count should now be 1 (just this new request).
        result = await check_user_rate_limit(
            redis, user_id="u1", max_requests=3, window_seconds=1
        )
        assert result is True

    async def test_get_user_request_count_returns_zero_initially(
        self, redis: FakeRedis
    ) -> None:
        count = await get_user_request_count(redis, user_id="u1")
        assert count == 0

    async def test_get_user_request_count_reflects_recorded_requests(
        self, redis: FakeRedis
    ) -> None:
        for _ in range(4):
            await check_user_rate_limit(redis, user_id="u1", max_requests=10)
        count = await get_user_request_count(redis, user_id="u1")
        assert count == 4


class TestGlobalRateLimit:
    async def test_first_request_is_allowed(self, redis: FakeRedis) -> None:
        result = await check_global_rate_limit(redis, max_rps=10)
        assert result is True

    async def test_requests_within_global_limit_are_allowed(self, redis: FakeRedis) -> None:
        for _ in range(5):
            ok = await check_global_rate_limit(redis, max_rps=10)
            assert ok is True

    async def test_global_limit_rejects_excess_requests(self, redis: FakeRedis) -> None:
        for _ in range(3):
            await check_global_rate_limit(redis, max_rps=3, window_seconds=60)
        result = await check_global_rate_limit(redis, max_rps=3, window_seconds=60)
        assert result is False

    async def test_global_key_exists_after_request(self, redis: FakeRedis) -> None:
        await check_global_rate_limit(redis, max_rps=10)
        assert await redis.exists(_KEY_GLOBAL) == 1

    async def test_global_key_has_ttl(self, redis: FakeRedis) -> None:
        await check_global_rate_limit(redis, max_rps=10, window_seconds=5)
        ttl = await redis.ttl(_KEY_GLOBAL)
        assert ttl > 0

    async def test_global_limit_resets_after_window(self, redis: FakeRedis) -> None:
        # Fill the 1-second window.
        for _ in range(2):
            await check_global_rate_limit(redis, max_rps=2, window_seconds=1)
        await asyncio.sleep(1.1)
        # After the window, a fresh request should be allowed.
        result = await check_global_rate_limit(redis, max_rps=2, window_seconds=1)
        assert result is True
