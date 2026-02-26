"""Async SQLAlchemy engine and session factory.

Usage
-----
At application startup (after settings are loaded):

    from tdbot.db.engine import create_engine, get_async_session

    engine = create_engine(settings)
    async with get_async_session(engine) as session:
        result = await session.execute(select(User))

"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from tdbot.config import Settings

# Cache sessionmaker instances per engine to avoid re-creating the factory on
# every call.  Keyed by engine id() so distinct engines get their own factory.
_session_factories: dict[int, async_sessionmaker[AsyncSession]] = {}


def create_engine(settings: Settings) -> AsyncEngine:
    """Create and return an async SQLAlchemy engine configured from *settings*."""
    dsn = settings.database_url.get_secret_value()
    return create_async_engine(
        dsn,
        pool_size=settings.database_pool_min,
        max_overflow=settings.database_pool_max - settings.database_pool_min,
        echo=False,
        future=True,
    )


def _get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return the cached sessionmaker for *engine*, creating it on first call."""
    key = id(engine)
    factory = _session_factories.get(key)
    if factory is None:
        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        _session_factories[key] = factory
    return factory


@asynccontextmanager
async def get_async_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a transactional :class:`AsyncSession`."""
    factory = _get_session_factory(engine)
    async with factory() as session, session.begin():
        yield session
