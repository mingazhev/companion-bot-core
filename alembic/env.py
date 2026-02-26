"""Alembic environment configuration with async SQLAlchemy support.

Migrations run against the database URL taken from the DATABASE_URL
environment variable.  The URL in alembic.ini is only a placeholder
and is always overridden here.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig
from typing import TYPE_CHECKING

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

# Import metadata so Alembic can auto-generate migrations.
from tdbot.db.models import Base

# ---------------------------------------------------------------------------
# Alembic Config object
# ---------------------------------------------------------------------------

config = context.config

# Wire stdlib logging from the alembic.ini [loggers] section.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Helper: resolve database URL
# ---------------------------------------------------------------------------


def _get_url() -> str:
    """Return the database URL from the environment, overriding alembic.ini."""
    url = os.environ.get(
        "DATABASE_URL",
        config.get_main_option("sqlalchemy.url", "") or "",
    )
    if not url:
        raise RuntimeError(
            "Database URL not configured. "
            "Set the DATABASE_URL environment variable."
        )
    # Ensure we use the asyncpg driver for migrations as well.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


# ---------------------------------------------------------------------------
# Offline migrations (generate SQL script without a live connection)
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL DDL to stdout without requiring a database connection.
    """
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations (run against a live database connection)
# ---------------------------------------------------------------------------


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations via a sync proxy connection."""
    engine = create_async_engine(_get_url(), poolclass=pool.NullPool)
    async with engine.connect() as conn:
        await conn.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration mode (called by Alembic CLI)."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
