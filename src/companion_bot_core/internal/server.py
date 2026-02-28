"""aiohttp application factory for the internal HTTP service.

Usage
-----
Build the app, wire routes, and run it with ``aiohttp.web.AppRunner`` alongside
the Telegram bot polling loop::

    app = build_internal_app(redis)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, settings.internal_server_host, settings.internal_server_port)
    await site.start()
    ...
    await runner.cleanup()

Endpoints
---------
POST /internal/refine/{user_id}
    Enqueue a prompt-refinement job.

POST /internal/detect-change
    Classify configuration-change intent.

GET /metrics
    Prometheus metrics in text exposition format.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from companion_bot_core.internal.routes import REDIS_KEY, handle_detect_change, handle_refine
from companion_bot_core.metrics import INTERNAL_REQUEST_LATENCY, INTERNAL_REQUESTS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from redis.asyncio import Redis


@web.middleware
async def _metrics_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Record per-endpoint request counts and latency for /internal/* routes."""
    # Skip the /metrics endpoint itself to avoid self-referential noise.
    if request.path == "/metrics":
        return await handler(request)

    # Derive a stable label from the resource's URL pattern so that
    # /internal/refine/{user_id} is always labelled "refine" regardless of
    # the actual user UUID in the path.
    try:
        canonical: str = request.match_info.route.resource.canonical  # type: ignore[union-attr]
        segments = [s for s in canonical.split("/") if s and not s.startswith("{")]
        endpoint = segments[-1] if segments else canonical
    except (AttributeError, IndexError):
        endpoint = request.path

    start = time.perf_counter()
    status_label = "success"
    try:
        response = await handler(request)
        if response.status >= 400:
            status_label = "error"
        return response
    except Exception:
        status_label = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        INTERNAL_REQUESTS.labels(endpoint=endpoint, status=status_label).inc()
        INTERNAL_REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)


async def _handle_metrics(_request: web.Request) -> web.Response:
    """GET /metrics — return Prometheus text exposition."""
    body: bytes = generate_latest()
    return web.Response(body=body, headers={"Content-Type": CONTENT_TYPE_LATEST})


def build_internal_app(redis: Redis) -> web.Application:
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
    app = web.Application(middlewares=[_metrics_middleware])
    app[REDIS_KEY] = redis

    app.router.add_post("/internal/refine/{user_id}", handle_refine)
    app.router.add_post("/internal/detect-change", handle_detect_change)
    app.router.add_get("/metrics", _handle_metrics)

    return app
