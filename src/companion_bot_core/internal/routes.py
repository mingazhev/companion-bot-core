"""aiohttp route handlers for the internal HTTP service.

These handlers are pure async functions; all shared state is accessed through
``request.app`` so they can be tested without standing up a real Redis or
running the full application.

Endpoints
---------
POST /internal/refine/{user_id}
    Enqueue a prompt-refinement job for the specified user.

POST /internal/detect-change
    Classify configuration-change intent in a user message and return the
    DetectionResult as JSON.

GET /internal/analytics/overview
    Aggregate engagement metrics over a configurable window.

GET /internal/analytics/users/{user_id}
    Per-user engagement profile.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from aiohttp import web
from pydantic import ValidationError

from companion_bot_core.behavior.detector import classify
from companion_bot_core.db.engine import get_async_session
from companion_bot_core.internal.analytics import get_analytics_overview, get_user_analytics
from companion_bot_core.internal.schemas import DetectChangeRequest, RefineRequest, RefineResponse
from companion_bot_core.logging_config import get_logger
from companion_bot_core.redis.queues import enqueue_refinement_job

log = get_logger(__name__)

# Type-safe application keys for shared resources.
# Using web.AppKey avoids the NotAppKeyWarning introduced in aiohttp 3.9+.
REDIS_KEY: web.AppKey[Any] = web.AppKey("redis")
ENGINE_KEY: web.AppKey[Any] = web.AppKey("engine")


async def handle_refine(request: web.Request) -> web.Response:
    """POST /internal/refine/{user_id} — enqueue a refinement job.

    Path parameters
    ---------------
    user_id : str
        Must be a valid UUID string.

    Request body (optional JSON)
    ----------------------------
    trigger : str
        Source label stored in the job payload.  Defaults to "internal_api".

    Responses
    ---------
    202 Accepted
        ``{"queued": true, "user_id": "...", "queue_length": N}``
    400 Bad Request
        ``{"error": "..."}`` when ``user_id`` is not a valid UUID or the body
        is malformed.
    """
    user_id_str: str = request.match_info["user_id"]
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return web.json_response(
            {"error": "invalid user_id: not a valid UUID"},
            status=400,
        )

    body: dict[str, Any] = {}
    # Read the body when present; Content-Length may be absent for chunked transfers.
    try:
        raw_bytes = await request.read()
    except Exception as exc:
        log.warning("refine_read_body_failed", error=str(exc))
        return web.json_response({"error": "failed to read request body"}, status=400)
    if raw_bytes:
        try:
            raw = json.loads(raw_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("refine_invalid_json", error=str(exc))
            return web.json_response({"error": "request body is not valid JSON"}, status=400)
        if not isinstance(raw, dict):
            return web.json_response(
                {"error": "request body must be a JSON object"}, status=400
            )
        body = raw

    try:
        req = RefineRequest(**body)
    except ValidationError as exc:
        return web.json_response({"error": exc.errors()}, status=400)

    redis = request.app[REDIS_KEY]

    # Acquire the shared dedup guard to prevent duplicate in-flight refinements.
    guard_key = f"refinement:pending:{user_id}"
    acquired = await redis.set(guard_key, "1", nx=True, ex=600)
    if not acquired:
        return web.json_response(
            {"queued": False, "user_id": str(user_id), "reason": "refinement already in progress"},
            status=409,
        )

    try:
        queue_length = await enqueue_refinement_job(
            redis,
            str(user_id),
            {"trigger": req.trigger},
        )
    except Exception:  # noqa: BLE001
        try:
            await redis.delete(guard_key)
        except Exception:  # noqa: BLE001
            log.warning("refine_guard_cleanup_failed", user_id=str(user_id))
        return web.json_response({"error": "failed to enqueue refinement job"}, status=500)

    resp = RefineResponse(
        queued=True,
        user_id=str(user_id),
        queue_length=queue_length,
    )
    return web.json_response(resp.model_dump(), status=202)


async def handle_detect_change(request: web.Request) -> web.Response:
    """POST /internal/detect-change — classify configuration intent.

    Request body (required JSON)
    ----------------------------
    text : str
        User message text to classify.  Must be non-empty.

    Responses
    ---------
    200 OK
        Full ``DetectionResult`` serialised as JSON.
    400 Bad Request
        ``{"error": "..."}`` when the body is missing, not valid JSON, or fails
        schema validation.
    """
    # Read body unconditionally; Content-Length may be absent for chunked transfers.
    try:
        raw_bytes = await request.read()
    except Exception as exc:
        log.warning("detect_change_read_body_failed", error=str(exc))
        return web.json_response({"error": "failed to read request body"}, status=400)
    if not raw_bytes:
        return web.json_response({"error": "request body is required"}, status=400)

    try:
        raw = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("detect_change_invalid_json", error=str(exc))
        return web.json_response({"error": "request body is not valid JSON"}, status=400)

    if not isinstance(raw, dict):
        return web.json_response(
            {"error": "request body must be a JSON object"}, status=400
        )

    try:
        req = DetectChangeRequest(**raw)
    except ValidationError as exc:
        return web.json_response({"error": exc.errors()}, status=400)

    result = classify(req.text)
    return web.json_response(result.model_dump())


async def handle_analytics_overview(request: web.Request) -> web.Response:
    """GET /internal/analytics/overview — aggregate engagement metrics.

    Query parameters
    ----------------
    days : int, optional
        Look-back window in days.  Defaults to 7.

    Responses
    ---------
    200 OK
        JSON with active_users, total_sessions, avg_session_messages, etc.
    400 Bad Request
        When ``days`` is not a valid positive integer.
    500 Internal Server Error
        When the database engine is not configured.
    """
    engine = request.app.get(ENGINE_KEY)
    if engine is None:
        return web.json_response(
            {"error": "analytics not available: no database engine configured"},
            status=500,
        )

    days_str = request.query.get("days", "7")
    try:
        days = int(days_str)
        if days < 1:
            raise ValueError  # noqa: TRY301
    except ValueError:
        return web.json_response(
            {"error": "days must be a positive integer"},
            status=400,
        )

    async with get_async_session(engine) as session:
        overview = await get_analytics_overview(session, days=days)

    return web.json_response(overview)


async def handle_analytics_user(request: web.Request) -> web.Response:
    """GET /internal/analytics/users/{user_id} — per-user engagement profile.

    Path parameters
    ---------------
    user_id : str
        Must be a valid UUID string.

    Query parameters
    ----------------
    days : int, optional
        Look-back window in days.  Defaults to 30.

    Responses
    ---------
    200 OK
        JSON with per-user metrics.
    400 Bad Request
        When ``user_id`` is not valid or ``days`` is not a positive integer.
    500 Internal Server Error
        When the database engine is not configured.
    """
    engine = request.app.get(ENGINE_KEY)
    if engine is None:
        return web.json_response(
            {"error": "analytics not available: no database engine configured"},
            status=500,
        )

    user_id_str: str = request.match_info["user_id"]
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return web.json_response(
            {"error": "invalid user_id: not a valid UUID"},
            status=400,
        )

    days_str = request.query.get("days", "30")
    try:
        days = int(days_str)
        if days < 1:
            raise ValueError  # noqa: TRY301
    except ValueError:
        return web.json_response(
            {"error": "days must be a positive integer"},
            status=400,
        )

    async with get_async_session(engine) as session:
        profile = await get_user_analytics(session, user_id, days=days)

    return web.json_response(profile)
