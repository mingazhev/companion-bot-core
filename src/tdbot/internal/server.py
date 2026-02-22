"""aiohttp application factory for the internal HTTP service.

Usage
-----
Build the app, wire routes, and run it with ``aiohttp.web.AppRunner`` alongside
the Telegram bot polling loop::

    app = build_internal_app(redis)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", settings.internal_server_port)
    await site.start()
    ...
    await runner.cleanup()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

from tdbot.internal.routes import REDIS_KEY, handle_detect_change, handle_refine

if TYPE_CHECKING:
    from redis.asyncio import Redis


def build_internal_app(redis: Redis[str]) -> web.Application:  # type: ignore[type-arg]
    """Create and return the internal aiohttp :class:`web.Application`.

    The ``redis`` client is stored on the application object so route handlers
    can access it without global state.

    Parameters
    ----------
    redis:
        An already-connected async Redis client.

    Returns
    -------
    web.Application
        The configured application, ready to be passed to ``AppRunner``.
    """
    app = web.Application()
    app[REDIS_KEY] = redis

    app.router.add_post("/internal/refine/{user_id}", handle_refine)
    app.router.add_post("/internal/detect-change", handle_detect_change)

    return app
