"""Application entry point.

Run with:
    python -m tdbot.main
or (after installing the package):
    tdbot

Polling mode is used when ``TELEGRAM_WEBHOOK_HOST`` is empty (default for
local development).  Set a non-empty ``TELEGRAM_WEBHOOK_HOST`` to switch to
webhook mode (webhook server setup is left to the deployment layer).

The internal HTTP service (``/internal/refine/{user_id}`` and
``/internal/detect-change``) is always started on
``INTERNAL_SERVER_HOST:INTERNAL_SERVER_PORT`` (default 127.0.0.1:8080).
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from aiohttp import web

from tdbot.bot.app import build_bot, build_dispatcher
from tdbot.config import get_settings
from tdbot.db.engine import create_engine, get_async_session
from tdbot.dev.fake_client import FakeChatAPIClient
from tdbot.inference.client import ChatAPIClient
from tdbot.internal.server import build_internal_app
from tdbot.logging_config import configure_logging, get_logger
from tdbot.privacy.field_encryption import FieldEncryptor
from tdbot.privacy.ttl_sweeper import sweep_expired_messages
from tdbot.prompt.postgres_store import PostgresSnapshotStore
from tdbot.prompt.snapshot_store import InMemorySnapshotStore, SnapshotStore
from tdbot.redis.client import close_redis_pool, create_redis_pool
from tdbot.refinement.worker import run_worker

if TYPE_CHECKING:
    from redis.asyncio import Redis


async def _run() -> None:
    settings = get_settings()
    configure_logging(settings)
    log = get_logger(__name__)

    log.info(
        "tdbot starting",
        service=settings.service_name,
        environment=settings.environment,
        chat_model=settings.chat_model,
        mode="polling" if not settings.telegram_webhook_host else "webhook",
        internal_host=settings.internal_server_host,
        internal_port=settings.internal_server_port,
    )

    engine = create_engine(settings)

    # Track resources for cleanup — all optional until initialized.
    redis: Redis | None = None
    chat_client: ChatAPIClient | None = None
    refinement_client: ChatAPIClient | None = None
    runner: web.AppRunner | None = None
    worker_task: asyncio.Task[None] | None = None
    sweeper_task: asyncio.Task[None] | None = None

    async def _run_ttl_sweeper() -> None:
        """Periodically delete expired conversation_messages rows."""
        while True:
            try:
                async with get_async_session(engine) as session:
                    deleted = await sweep_expired_messages(session)
                log.info("ttl_sweep_done", deleted=deleted)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ttl_sweep_error")
            await asyncio.sleep(3600)  # wait one hour between sweeps

    try:
        redis = await create_redis_pool(settings)
        snapshot_store: SnapshotStore
        if settings.use_fake_adapters:
            snapshot_store = InMemorySnapshotStore()
        else:
            snapshot_store = PostgresSnapshotStore(engine=engine, redis=redis)

        encryptor = FieldEncryptor.from_settings(settings)
        if encryptor.is_enabled:
            log.info("field_encryption_enabled")

        if settings.use_fake_adapters:
            log.warning(
                "fake_adapters_enabled",
                reason="USE_FAKE_ADAPTERS=true — no real model API calls will be made",
            )
            chat_client = FakeChatAPIClient(model=settings.chat_model)
            refinement_client = FakeChatAPIClient(model=settings.refinement_model)
        else:
            chat_client = ChatAPIClient(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.chat_model,
                base_url=settings.openai_base_url,
            )
            refinement_client = ChatAPIClient(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.refinement_model,
                base_url=settings.openai_base_url,
            )

        # Start the internal HTTP service.
        internal_app = build_internal_app(redis)
        runner = web.AppRunner(internal_app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            settings.internal_server_host,
            settings.internal_server_port,
        )
        await site.start()
        log.info(
            "internal_server_started",
            host=settings.internal_server_host,
            port=settings.internal_server_port,
        )

        bot = build_bot(settings)
        dp = build_dispatcher(
            settings,
            engine=engine,
            redis=redis,
            snapshot_store=snapshot_store,
            chat_client=chat_client,
            encryptor=encryptor,
        )

        worker_task = asyncio.create_task(
            run_worker(
                redis=redis,
                snapshot_store=snapshot_store,
                chat_client=refinement_client,
                engine=engine,
                encryptor=encryptor,
            ),
            name="refinement_worker",
        )
        sweeper_task = asyncio.create_task(
            _run_ttl_sweeper(),
            name="ttl_sweeper",
        )

        if not settings.telegram_webhook_host:
            log.info("starting_polling")
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        else:
            log.info(
                "webhook_mode",
                host=settings.telegram_webhook_host,
                path=settings.telegram_webhook_path,
            )
            # Webhook wiring (web server setup) is handled by the deployment layer.
            # For now, raise to signal incomplete setup.
            raise NotImplementedError(
                "Webhook mode requires a web server integration (e.g., aiohttp). "
                "Set TELEGRAM_WEBHOOK_HOST='' to use polling mode."
            )
    finally:
        if worker_task is not None:
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
        if sweeper_task is not None:
            sweeper_task.cancel()
            await asyncio.gather(sweeper_task, return_exceptions=True)
        if runner is not None:
            await runner.cleanup()
        if chat_client is not None:
            await chat_client.close()
        if refinement_client is not None:
            await refinement_client.close()
        if redis is not None:
            await close_redis_pool(redis)
        await engine.dispose()
        log.info("tdbot stopped")


def main() -> None:
    """Synchronous entry-point for the ``tdbot`` console script."""
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
