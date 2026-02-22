"""Unit tests for Telegram update idempotency keys."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import fakeredis
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from tdbot.redis.idempotency import (
    _PREFIX,
    DEFAULT_TTL_SECONDS,
    clear_update_key,
    is_update_seen,
    mark_update_seen,
)

FakeRedis = fakeredis.FakeAsyncRedis


@pytest_asyncio.fixture
async def redis() -> AsyncGenerator[FakeRedis, None]:
    client = FakeRedis(decode_responses=True)
    yield client
    await client.aclose()  # type: ignore[attr-defined]


class TestMarkUpdateSeen:
    async def test_first_call_returns_true(self, redis: FakeRedis) -> None:
        result = await mark_update_seen(redis, update_id=1001)
        assert result is True

    async def test_second_call_with_same_id_returns_false(self, redis: FakeRedis) -> None:
        await mark_update_seen(redis, update_id=1001)
        result = await mark_update_seen(redis, update_id=1001)
        assert result is False

    async def test_different_update_ids_are_independent(self, redis: FakeRedis) -> None:
        assert await mark_update_seen(redis, update_id=1) is True
        assert await mark_update_seen(redis, update_id=2) is True
        assert await mark_update_seen(redis, update_id=1) is False
        assert await mark_update_seen(redis, update_id=2) is False

    async def test_key_uses_expected_prefix(self, redis: FakeRedis) -> None:
        await mark_update_seen(redis, update_id=42)
        keys = await redis.keys(f"{_PREFIX}*")
        assert f"{_PREFIX}42" in keys

    async def test_key_has_ttl_set(self, redis: FakeRedis) -> None:
        await mark_update_seen(redis, update_id=99)
        ttl = await redis.ttl(f"{_PREFIX}99")
        assert 0 < ttl <= DEFAULT_TTL_SECONDS

    async def test_custom_ttl_is_applied(self, redis: FakeRedis) -> None:
        await mark_update_seen(redis, update_id=55, ttl_seconds=120)
        ttl = await redis.ttl(f"{_PREFIX}55")
        assert 0 < ttl <= 120

    async def test_key_expires_after_ttl(self, redis: FakeRedis) -> None:
        await mark_update_seen(redis, update_id=77, ttl_seconds=1)
        await asyncio.sleep(1.1)
        # After expiry, the same update_id is treated as new.
        result = await mark_update_seen(redis, update_id=77, ttl_seconds=1)
        assert result is True

    async def test_large_update_id_is_handled(self, redis: FakeRedis) -> None:
        large_id = 2**31 - 1  # max signed 32-bit int
        result = await mark_update_seen(redis, update_id=large_id)
        assert result is True
        assert await mark_update_seen(redis, update_id=large_id) is False


class TestIsUpdateSeen:
    async def test_returns_false_before_marking(self, redis: FakeRedis) -> None:
        result = await is_update_seen(redis, update_id=200)
        assert result is False

    async def test_returns_true_after_marking(self, redis: FakeRedis) -> None:
        await mark_update_seen(redis, update_id=200)
        result = await is_update_seen(redis, update_id=200)
        assert result is True

    async def test_does_not_consume_idempotency_token(self, redis: FakeRedis) -> None:
        # is_update_seen is read-only; mark_update_seen should still succeed after it.
        await is_update_seen(redis, update_id=300)
        result = await mark_update_seen(redis, update_id=300)
        assert result is True


class TestClearUpdateKey:
    async def test_removes_existing_key(self, redis: FakeRedis) -> None:
        await mark_update_seen(redis, update_id=500)
        await clear_update_key(redis, update_id=500)
        assert await is_update_seen(redis, update_id=500) is False

    async def test_no_error_when_key_absent(self, redis: FakeRedis) -> None:
        # Should not raise.
        await clear_update_key(redis, update_id=999)

    async def test_mark_allowed_after_clear(self, redis: FakeRedis) -> None:
        await mark_update_seen(redis, update_id=600)
        await clear_update_key(redis, update_id=600)
        result = await mark_update_seen(redis, update_id=600)
        assert result is True
