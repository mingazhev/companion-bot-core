"""Application entry point.

Run with:
    python -m tdbot.main
or (after installing the package):
    tdbot
"""

from __future__ import annotations

import asyncio
import sys

from tdbot.config import get_settings
from tdbot.logging_config import configure_logging, get_logger


async def _run() -> None:
    settings = get_settings()
    configure_logging(settings)
    log = get_logger(__name__)
    log.info(
        "tdbot starting",
        service=settings.service_name,
        environment=settings.environment,
        chat_model=settings.chat_model,
    )
    # Placeholder: actual bot wiring is implemented in Task 4 (Telegram ingress).
    log.info("Bot is not yet wired to Telegram — skipping startup (stub)")


def main() -> None:
    """Synchronous entry-point for the `tdbot` console script."""
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
