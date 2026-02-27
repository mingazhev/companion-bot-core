"""Unit tests for IngressMiddleware — idempotency, rate limits, user provisioning."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis
import pytest
from pydantic import SecretStr

from tdbot.bot.middleware import IngressMiddleware
from tdbot.config import Settings
from tdbot.db.models import User

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "telegram_bot_token": SecretStr("1234567890:AAFakeToken"),
        "database_url": SecretStr("postgresql+asyncpg://u:p@localhost/db"),
        "redis_url": SecretStr("redis://localhost:6379/0"),
        "openai_api_key": SecretStr("sk-test"),
        "rate_limit_messages_per_minute": 20,
        "rate_limit_global_rps": 100,
        "encrypt_sensitive_fields": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_update(update_id: int, telegram_user_id: int = 123) -> MagicMock:
    """Build a minimal fake aiogram Update with a message from a user."""
    tg_user = MagicMock()
    tg_user.id = telegram_user_id

    message = MagicMock()
    message.from_user = tg_user

    update = MagicMock()
    update.update_id = update_id
    update.message = message
    update.callback_query = None
    return update


def _make_db_user(telegram_user_id: int = 123) -> User:
    user = User(telegram_user_id=telegram_user_id)
    user.id = uuid.uuid4()
    return user


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_duplicate_update_is_dropped() -> None:
    """Second call with the same update_id must not invoke the handler."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    settings = _make_settings()
    engine = MagicMock()
    middleware = IngressMiddleware(settings=settings, engine=engine, redis=redis)

    handler = AsyncMock(return_value="ok")
    update = _make_update(update_id=1)

    # First call — should pass through
    with patch("tdbot.bot.middleware.get_async_session") as mock_session_cm, patch(
        "tdbot.bot.middleware.get_or_create_user", return_value=_make_db_user()
    ):
        mock_session = AsyncMock()
        mock_session.info = {}
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cm.return_value = mock_session

        await middleware(handler, update, {})

    # Second call — same update_id, handler must NOT be called again
    handler.reset_mock()
    result2 = await middleware(handler, update, {})

    assert result2 is None
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_new_update_calls_handler() -> None:
    """An unseen update_id should invoke the handler."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    settings = _make_settings()
    engine = MagicMock()
    middleware = IngressMiddleware(settings=settings, engine=engine, redis=redis)

    db_user = _make_db_user()
    handler = AsyncMock(return_value="response")

    with patch("tdbot.bot.middleware.get_async_session") as mock_session_cm, patch(
        "tdbot.bot.middleware.get_or_create_user", return_value=db_user
    ):
        mock_session = AsyncMock()
        mock_session.info = {}
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cm.return_value = mock_session

        data: dict[str, Any] = {}
        await middleware(handler, _make_update(update_id=42), data)

    handler.assert_called_once()
    assert data["db_user"] is db_user


@pytest.mark.asyncio
async def test_user_provisioned_in_data() -> None:
    """Middleware must inject db_user and tg_user into the data dict."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    settings = _make_settings()
    engine = MagicMock()
    middleware = IngressMiddleware(settings=settings, engine=engine, redis=redis)

    db_user = _make_db_user(telegram_user_id=777)
    handler = AsyncMock(return_value=None)

    with patch("tdbot.bot.middleware.get_async_session") as mock_session_cm, patch(
        "tdbot.bot.middleware.get_or_create_user", return_value=db_user
    ):
        mock_session = AsyncMock()
        mock_session.info = {}
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cm.return_value = mock_session

        data: dict[str, Any] = {}
        await middleware(handler, _make_update(update_id=100, telegram_user_id=777), data)

    assert data["db_user"] is db_user
    assert data["tg_user"].id == 777


@pytest.mark.asyncio
async def test_update_without_user_passes_through() -> None:
    """Updates without an identifiable user (e.g. channel posts) pass to handler."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    settings = _make_settings()
    engine = MagicMock()
    middleware = IngressMiddleware(settings=settings, engine=engine, redis=redis)

    # Build an update with no from_user
    update = MagicMock()
    update.update_id = 200
    update.message = None
    update.callback_query = None

    handler = AsyncMock(return_value="through")
    data: dict[str, Any] = {}
    await middleware(handler, update, data)

    handler.assert_called_once()
    assert "db_user" not in data


@pytest.mark.asyncio
async def test_per_user_rate_limit_drops_excess() -> None:
    """When a user exceeds their per-minute message cap, excess updates are dropped."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    # Very tight cap — only 1 message per minute
    settings = _make_settings(rate_limit_messages_per_minute=1)
    engine = MagicMock()
    middleware = IngressMiddleware(settings=settings, engine=engine, redis=redis)

    db_user = _make_db_user()
    handler = AsyncMock(return_value=None)

    async def run_update(update_id: int) -> Any:
        with patch("tdbot.bot.middleware.get_async_session") as mock_cm, patch(
            "tdbot.bot.middleware.get_or_create_user", return_value=db_user
        ):
            sess = AsyncMock()
            sess.info = {}
            sess.__aenter__ = AsyncMock(return_value=sess)
            sess.__aexit__ = AsyncMock(return_value=False)
            mock_cm.return_value = sess
            return await middleware(handler, _make_update(update_id=update_id), {})

    # First message — should pass (count=1, limit=1)
    await run_update(update_id=300)
    # Second message — should be dropped (count=2, limit=1)
    handler.reset_mock()
    result = await run_update(update_id=301)
    assert result is None
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_global_rate_limit_drops_excess() -> None:
    """When the global RPS cap is saturated, excess updates are dropped."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    # Cap of 1 RPS so the second request in the same second is dropped
    settings = _make_settings(rate_limit_global_rps=1)
    engine = MagicMock()
    middleware = IngressMiddleware(settings=settings, engine=engine, redis=redis)

    db_user = _make_db_user()
    handler = AsyncMock(return_value=None)

    async def run_update(update_id: int) -> Any:
        with patch("tdbot.bot.middleware.get_async_session") as mock_cm, patch(
            "tdbot.bot.middleware.get_or_create_user", return_value=db_user
        ):
            sess = AsyncMock()
            sess.info = {}
            sess.__aenter__ = AsyncMock(return_value=sess)
            sess.__aexit__ = AsyncMock(return_value=False)
            mock_cm.return_value = sess
            return await middleware(handler, _make_update(update_id=update_id), {})

    # First request — within global cap
    await run_update(update_id=500)
    # All subsequent requests in the same time window exceed the cap and are dropped
    handler.reset_mock()
    results = [await run_update(update_id=500 + i + 1) for i in range(5)]
    assert all(r is None for r in results)
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_deferred_redis_flush_failure_preserves_idempotency_key() -> None:
    """If deferred Redis flush fails after a successful handler + commit,
    the idempotency key must NOT be cleared — otherwise Telegram retries
    would duplicate an already-committed update."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    settings = _make_settings()
    engine = MagicMock()
    middleware = IngressMiddleware(settings=settings, engine=engine, redis=redis)

    handler = AsyncMock(return_value="ok")
    update = _make_update(update_id=900)

    flush_mock = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
    with patch("tdbot.bot.middleware.get_async_session") as mock_session_cm, patch(
        "tdbot.bot.middleware.get_or_create_user", return_value=_make_db_user()
    ), patch(
        "tdbot.bot.middleware.extract_deferred_redis_writes",
        return_value=[("prompt:active:test", "snap-id")],
    ), patch(
        "tdbot.bot.middleware.flush_deferred_redis_writes",
        new=flush_mock,
    ):
        mock_session = AsyncMock()
        mock_session.info = {}
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cm.return_value = mock_session

        # Should NOT raise — the flush error is caught and logged.
        result = await middleware(handler, update, {})

    assert result == "ok"
    handler.assert_called_once()
    # The idempotency key must still be present so retries are rejected.
    assert await redis.exists("idempotency:update:900") == 1
    # All 3 retry attempts must be made before giving up.
    assert flush_mock.await_count == 3
