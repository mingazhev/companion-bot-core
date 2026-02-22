"""Application entry point.

Run with:
    python -m tdbot.main
or (after installing the package):
    tdbot

Polling mode is used when ``TELEGRAM_WEBHOOK_HOST`` is empty (default for
local development).  Set a non-empty ``TELEGRAM_WEBHOOK_HOST`` to switch to
webhook mode (webhook server setup is left to the deployment layer).
"""

from __future__ import annotations

import asyncio
import sys

from tdbot.bot.app import build_bot, build_dispatcher
from tdbot.config import get_settings
from tdbot.db.engine import create_engine
from tdbot.logging_config import configure_logging, get_logger
from tdbot.redis.client import close_redis_pool, create_redis_pool


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
    )

    engine = create_engine(settings)
    redis = await create_redis_pool(settings)

    try:
        bot = build_bot(settings)
        dp = build_dispatcher(settings, engine=engine, redis=redis)

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
