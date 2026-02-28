"""Unit tests for the Redis prompt context cache."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import fakeredis
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from companion_bot_core.redis.prompt_cache import (
    _PREFIX,
    DEFAULT_TTL_SECONDS,
    cache_prompt,
    extend_prompt_cache_ttl,
    get_cached_prompt,
    invalidate_prompt_cache,
)

FakeRedis = fakeredis.FakeAsyncRedis


@pytest_asyncio.fixture
async def redis() -> AsyncGenerator[FakeRedis, None]:
    client = FakeRedis(decode_responses=True)
    yield client
    await client.aclose()  # type: ignore[attr-defined]


_SAMPLE_PROMPT: dict[str, object] = {
    "system_prompt": "You are a friendly assistant.",
    "persona_name": "Alex",
    "skill_prompts": {"greeting": "Greet warmly."},
    "version": 3,
}


class TestCachePrompt:
    async def test_stores_prompt_data(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="u1", prompt_data=_SAMPLE_PROMPT)
        raw = await redis.get(f"{_PREFIX}u1")
        assert raw is not None
        assert "friendly assistant" in raw

    async def test_stores_with_ttl(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="u1", prompt_data=_SAMPLE_PROMPT, ttl_seconds=120)
        ttl = await redis.ttl(f"{_PREFIX}u1")
        assert 0 < ttl <= 120

    async def test_uses_default_ttl(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="u1", prompt_data=_SAMPLE_PROMPT)
        ttl = await redis.ttl(f"{_PREFIX}u1")
        assert 0 < ttl <= DEFAULT_TTL_SECONDS

    async def test_overwrites_existing_entry(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="u1", prompt_data={"version": 1})
        await cache_prompt(redis, user_id="u1", prompt_data={"version": 2})
        result = await get_cached_prompt(redis, user_id="u1")
        assert result == {"version": 2}

    async def test_different_users_have_separate_keys(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="alice", prompt_data={"persona": "Alice"})
        await cache_prompt(redis, user_id="bob", prompt_data={"persona": "Bob"})
        alice_result = await get_cached_prompt(redis, user_id="alice")
        bob_result = await get_cached_prompt(redis, user_id="bob")
        assert alice_result == {"persona": "Alice"}
        assert bob_result == {"persona": "Bob"}


class TestGetCachedPrompt:
    async def test_returns_none_on_miss(self, redis: FakeRedis) -> None:
        result = await get_cached_prompt(redis, user_id="missing")
        assert result is None

    async def test_returns_stored_data(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="u1", prompt_data=_SAMPLE_PROMPT)
        result = await get_cached_prompt(redis, user_id="u1")
        assert result == _SAMPLE_PROMPT

    async def test_returns_none_after_expiry(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="u1", prompt_data=_SAMPLE_PROMPT, ttl_seconds=1)
        await asyncio.sleep(1.1)
        result = await get_cached_prompt(redis, user_id="u1")
        assert result is None

    async def test_preserves_nested_structures(self, redis: FakeRedis) -> None:
        data: dict[str, object] = {
            "system_prompt": "Hello",
            "skills": {"tone": "friendly", "lang": "en"},
            "flags": [1, 2, 3],
        }
        await cache_prompt(redis, user_id="u1", prompt_data=data)
        result = await get_cached_prompt(redis, user_id="u1")
        assert result == data


class TestInvalidatePromptCache:
    async def test_removes_existing_entry(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="u1", prompt_data=_SAMPLE_PROMPT)
        await invalidate_prompt_cache(redis, user_id="u1")
        result = await get_cached_prompt(redis, user_id="u1")
        assert result is None

    async def test_no_error_when_key_absent(self, redis: FakeRedis) -> None:
        # Should not raise.
        await invalidate_prompt_cache(redis, user_id="nonexistent")

    async def test_only_invalidates_target_user(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="alice", prompt_data={"v": 1})
        await cache_prompt(redis, user_id="bob", prompt_data={"v": 2})
        await invalidate_prompt_cache(redis, user_id="alice")
        assert await get_cached_prompt(redis, user_id="alice") is None
        assert await get_cached_prompt(redis, user_id="bob") == {"v": 2}


class TestExtendPromptCacheTtl:
    async def test_returns_true_when_key_exists(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="u1", prompt_data=_SAMPLE_PROMPT, ttl_seconds=60)
        result = await extend_prompt_cache_ttl(redis, user_id="u1", ttl_seconds=300)
        assert result is True

    async def test_returns_false_when_key_absent(self, redis: FakeRedis) -> None:
        result = await extend_prompt_cache_ttl(redis, user_id="missing", ttl_seconds=300)
        assert result is False

    async def test_ttl_is_updated(self, redis: FakeRedis) -> None:
        await cache_prompt(redis, user_id="u1", prompt_data=_SAMPLE_PROMPT, ttl_seconds=10)
        await extend_prompt_cache_ttl(redis, user_id="u1", ttl_seconds=500)
        ttl = await redis.ttl(f"{_PREFIX}u1")
        assert ttl > 10
