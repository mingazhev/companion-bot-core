"""Async SQLAlchemy engine and session factory.

Usage
-----
At application startup (after settings are loaded):

    from companion_bot_core.db.engine import create_engine, get_async_session

    engine = create_engine(settings)
    async with get_async_session(engine) as session:
        result = await session.execute(select(User))

"""

from __future__ import annotations

import weakref
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

    from companion_bot_core.config import Settings

# Cache sessionmaker instances per engine using weak references so that a
# disposed engine's entry is automatically evicted.  Using id() was unsafe
# because CPython can reuse the same integer for a new object after the
# original is GC'd, causing the wrong factory to be returned.
_session_factories: weakref.WeakKeyDictionary[
    AsyncEngine, async_sessionmaker[AsyncSession]
] = weakref.WeakKeyDictionary()


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
    factory = _session_factories.get(engine)
    if factory is None:
        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        _session_factories[engine] = factory
    return factory


@asynccontextmanager
async def get_async_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a transactional :class:`AsyncSession`."""
    factory = _get_session_factory(engine)
    async with factory() as session, session.begin():
        yield session
