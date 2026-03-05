"""Unit tests for conversation session tracking and analytics (3.2).

Tests cover:
- session_tracker.track_session: session creation, continuation, boundary
  detection, mood tracking, farewell marking
- analytics.get_analytics_overview: aggregate metrics
- analytics.get_user_analytics: per-user engagement profile
- Analytics HTTP endpoints: overview and per-user routes
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis
import pytest
from aiohttp.test_utils import TestClient, TestServer

from companion_bot_core.db.models import ConversationSession
from companion_bot_core.internal.analytics import get_analytics_overview, get_user_analytics
from companion_bot_core.internal.server import build_internal_app
from companion_bot_core.orchestrator.session_tracker import track_session

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_TC = TestClient[Any, Any]


# ---------------------------------------------------------------------------
# Fake in-memory DB session for track_session tests
# ---------------------------------------------------------------------------


class FakeAsyncSession:
    """Minimal async session stub backed by a list for track_session tests."""

    def __init__(self) -> None:
        self._store: list[ConversationSession] = []

    async def execute(self, stmt: Any) -> Any:
        """Simulate SQLAlchemy execute for SELECT ... ORDER BY ended_at DESC LIMIT 1."""
        user_id = None
        # Extract user_id from the statement's where clause.
        clauses: list[Any] = []
        if hasattr(stmt, "whereclause"):
            wc = stmt.whereclause
            clauses = list(wc.clauses) if hasattr(wc, "clauses") else [wc]
        for clause in clauses:
            if hasattr(clause, "right") and hasattr(clause.right, "value"):
                user_id = clause.right.value
                break

        matching = [
            s for s in self._store
            if s.user_id == user_id
        ]
        matching.sort(key=lambda s: s.ended_at, reverse=True)

        result = MagicMock()
        result.scalar_one_or_none.return_value = matching[0] if matching else None
        return result

    def add(self, obj: Any) -> None:
        self._store.append(obj)

    async def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# track_session
# ---------------------------------------------------------------------------


class TestTrackSession:
    @pytest.mark.asyncio
    async def test_creates_new_session_when_none_exists(self) -> None:
        db = FakeAsyncSession()
        user_id = uuid.uuid4()

        await track_session(db, user_id)  # type: ignore[arg-type]

        assert len(db._store) == 1
        s = db._store[0]
        assert s.user_id == user_id
        assert s.message_count == 1
        assert s.ended_with_farewell is False
        assert s.dominant_mood is None

    @pytest.mark.asyncio
    async def test_continues_existing_session_within_gap(self) -> None:
        db = FakeAsyncSession()
        user_id = uuid.uuid4()
        now = datetime.now(tz=UTC)

        existing = ConversationSession(
            user_id=user_id,
            started_at=now - timedelta(minutes=5),
            ended_at=now - timedelta(minutes=1),
            message_count=3,
            dominant_mood=None,
            ended_with_farewell=False,
        )
        db._store.append(existing)

        await track_session(db, user_id)  # type: ignore[arg-type]

        # Should NOT create a new session
        assert len(db._store) == 1
        assert existing.message_count == 4
        assert existing.ended_at >= now - timedelta(seconds=5)

    @pytest.mark.asyncio
    async def test_creates_new_session_after_gap_expires(self) -> None:
        db = FakeAsyncSession()
        user_id = uuid.uuid4()
        now = datetime.now(tz=UTC)

        old = ConversationSession(
            user_id=user_id,
            started_at=now - timedelta(minutes=60),
            ended_at=now - timedelta(minutes=35),
            message_count=5,
            dominant_mood="venting",
            ended_with_farewell=False,
        )
        db._store.append(old)

        await track_session(db, user_id)  # type: ignore[arg-type]

        assert len(db._store) == 2
        new = db._store[1]
        assert new.message_count == 1
        assert old.message_count == 5  # unchanged

    @pytest.mark.asyncio
    async def test_updates_dominant_mood_on_non_neutral(self) -> None:
        db = FakeAsyncSession()
        user_id = uuid.uuid4()

        await track_session(db, user_id, emotion_mode="neutral")  # type: ignore[arg-type]
        assert db._store[0].dominant_mood is None

        await track_session(db, user_id, emotion_mode="venting")  # type: ignore[arg-type]
        assert db._store[0].dominant_mood == "venting"

        await track_session(db, user_id, emotion_mode="task")  # type: ignore[arg-type]
        assert db._store[0].dominant_mood == "task"

    @pytest.mark.asyncio
    async def test_sets_farewell_flag(self) -> None:
        db = FakeAsyncSession()
        user_id = uuid.uuid4()

        await track_session(db, user_id)  # type: ignore[arg-type]
        assert db._store[0].ended_with_farewell is False

        await track_session(db, user_id, is_farewell=True)  # type: ignore[arg-type]
        assert db._store[0].ended_with_farewell is True

    @pytest.mark.asyncio
    async def test_new_session_with_farewell(self) -> None:
        db = FakeAsyncSession()
        user_id = uuid.uuid4()

        await track_session(  # type: ignore[arg-type]
            db, user_id, emotion_mode="farewell", is_farewell=True,
        )

        s = db._store[0]
        assert s.ended_with_farewell is True
        assert s.dominant_mood == "farewell"

    @pytest.mark.asyncio
    async def test_neutral_mood_does_not_overwrite(self) -> None:
        db = FakeAsyncSession()
        user_id = uuid.uuid4()

        await track_session(db, user_id, emotion_mode="venting")  # type: ignore[arg-type]
        assert db._store[0].dominant_mood == "venting"

        await track_session(db, user_id, emotion_mode="neutral")  # type: ignore[arg-type]
        assert db._store[0].dominant_mood == "venting"  # unchanged

    @pytest.mark.asyncio
    async def test_different_users_get_separate_sessions(self) -> None:
        db = FakeAsyncSession()
        user1 = uuid.uuid4()
        user2 = uuid.uuid4()

        await track_session(db, user1)  # type: ignore[arg-type]
        await track_session(db, user2)  # type: ignore[arg-type]

        assert len(db._store) == 2
        assert db._store[0].user_id == user1
        assert db._store[1].user_id == user2


# ---------------------------------------------------------------------------
# Analytics queries (using real SQLAlchemy ORM objects in-memory)
# ---------------------------------------------------------------------------


class FakeAnalyticsSession:
    """Minimal session stub for analytics query tests.

    Stores ConversationSession objects and supports the SQLAlchemy-style
    execute pattern used by analytics.py.
    """

    def __init__(self, sessions: list[ConversationSession]) -> None:
        self._sessions = sessions

    async def execute(self, stmt: Any) -> Any:
        """Delegate to a real in-memory filter. For simplicity, returns all."""
        result = MagicMock()
        # Analytics queries use scalar_one() or scalars().all()
        # We need to handle both patterns.
        if hasattr(stmt, 'is_select') and stmt.is_select:
            # For select queries returning full rows
            scalars_mock = MagicMock()
            scalars_mock.all.return_value = self._sessions
            result.scalars.return_value = scalars_mock
            result.scalar_one.return_value = self._compute_scalar(stmt)
        else:
            result.scalar_one.return_value = self._compute_scalar(stmt)
        return result

    def _compute_scalar(self, stmt: Any) -> Any:
        return len(self._sessions)


class TestAnalyticsOverview:
    @pytest.mark.asyncio
    async def test_empty_returns_zeros(self) -> None:
        session = AsyncMock()
        # All aggregate queries return 0
        result_mock = MagicMock()
        result_mock.scalar_one.return_value = 0
        session.execute = AsyncMock(return_value=result_mock)

        overview = await get_analytics_overview(session, days=7)

        assert overview["active_users"] == 0
        assert overview["total_sessions"] == 0
        assert overview["avg_session_messages"] == 0.0
        assert overview["farewell_rate"] == 0.0
        assert overview["return_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_returns_expected_keys(self) -> None:
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one.return_value = 0
        session.execute = AsyncMock(return_value=result_mock)

        overview = await get_analytics_overview(session, days=7)

        expected_keys = {
            "period_days", "active_users", "total_sessions",
            "avg_session_messages", "avg_session_duration_seconds",
            "farewell_rate", "return_rate",
        }
        assert set(overview.keys()) == expected_keys


class TestUserAnalytics:
    @pytest.mark.asyncio
    async def test_no_sessions_returns_zeros(self) -> None:
        session = AsyncMock()
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = []
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=result_mock)

        user_id = uuid.uuid4()
        profile = await get_user_analytics(session, user_id, days=30)

        assert profile["user_id"] == str(user_id)
        assert profile["total_sessions"] == 0
        assert profile["total_messages"] == 0
        assert profile["last_session_at"] is None
        assert profile["dominant_moods"] == {}

    @pytest.mark.asyncio
    async def test_with_sessions_computes_metrics(self) -> None:
        user_id = uuid.uuid4()
        now = datetime.now(tz=UTC)

        sessions = [
            ConversationSession(
                user_id=user_id,
                started_at=now - timedelta(hours=1),
                ended_at=now - timedelta(minutes=30),
                message_count=5,
                dominant_mood="venting",
                ended_with_farewell=True,
            ),
            ConversationSession(
                user_id=user_id,
                started_at=now - timedelta(days=1),
                ended_at=now - timedelta(days=1) + timedelta(minutes=10),
                message_count=3,
                dominant_mood="task",
                ended_with_farewell=False,
            ),
        ]

        session = AsyncMock()
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = sessions
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=result_mock)

        profile = await get_user_analytics(session, user_id, days=30)

        assert profile["total_sessions"] == 2
        assert profile["total_messages"] == 8
        assert profile["avg_session_messages"] == 4.0
        assert profile["farewell_rate"] == 0.5
        assert profile["dominant_moods"] == {"venting": 1, "task": 1}
        assert profile["last_session_at"] is not None

    @pytest.mark.asyncio
    async def test_mood_distribution_counts(self) -> None:
        user_id = uuid.uuid4()
        now = datetime.now(tz=UTC)

        sessions = [
            ConversationSession(
                user_id=user_id,
                started_at=now - timedelta(hours=i),
                ended_at=now - timedelta(hours=i) + timedelta(minutes=5),
                message_count=2,
                dominant_mood="venting" if i % 2 == 0 else "task",
                ended_with_farewell=False,
            )
            for i in range(4)
        ]

        session = AsyncMock()
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = sessions
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=result_mock)

        profile = await get_user_analytics(session, user_id, days=30)

        assert profile["dominant_moods"]["venting"] == 2
        assert profile["dominant_moods"]["task"] == 2


# ---------------------------------------------------------------------------
# Analytics HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
async def redis() -> AsyncGenerator[fakeredis.FakeRedis, None]:
    client: fakeredis.FakeRedis = fakeredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture()
async def analytics_client(redis: fakeredis.FakeRedis) -> AsyncGenerator[_TC, None]:
    # Build with no engine — analytics endpoints should return 500.
    app = build_internal_app(redis)  # type: ignore[arg-type]
    server = TestServer(app)
    tc: _TC = TestClient(server)
    await tc.start_server()
    yield tc
    await tc.close()


class TestAnalyticsOverviewEndpoint:
    async def test_no_engine_returns_500(
        self, analytics_client: _TC,
    ) -> None:
        resp = await analytics_client.get("/internal/analytics/overview")
        assert resp.status == 500
        body = await resp.json()
        assert "not available" in body["error"]

    async def test_invalid_days_returns_400(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        mock_engine = MagicMock()
        app = build_internal_app(redis, engine=mock_engine)  # type: ignore[arg-type]
        server = TestServer(app)
        tc: _TC = TestClient(server)
        await tc.start_server()
        try:
            resp = await tc.get("/internal/analytics/overview?days=abc")
            assert resp.status == 400
            body = await resp.json()
            assert "positive integer" in body["error"]
        finally:
            await tc.close()

    async def test_negative_days_returns_400(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        mock_engine = MagicMock()
        app = build_internal_app(redis, engine=mock_engine)  # type: ignore[arg-type]
        server = TestServer(app)
        tc: _TC = TestClient(server)
        await tc.start_server()
        try:
            resp = await tc.get("/internal/analytics/overview?days=-1")
            assert resp.status == 400
        finally:
            await tc.close()

    async def test_valid_request_returns_200(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        mock_overview = {
            "period_days": 7,
            "active_users": 10,
            "total_sessions": 25,
            "avg_session_messages": 4.5,
            "avg_session_duration_seconds": 300.0,
            "farewell_rate": 0.4,
            "return_rate": 0.6,
        }
        mock_engine = MagicMock()
        app = build_internal_app(redis, engine=mock_engine)  # type: ignore[arg-type]
        server = TestServer(app)
        tc: _TC = TestClient(server)
        await tc.start_server()
        try:
            with patch(
                "companion_bot_core.internal.routes.get_analytics_overview",
                return_value=mock_overview,
            ), patch(
                "companion_bot_core.internal.routes.get_async_session",
            ) as mock_session_ctx:
                mock_ctx = AsyncMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
                mock_session_ctx.return_value = mock_ctx

                resp = await tc.get("/internal/analytics/overview")
                assert resp.status == 200
                body = await resp.json()
                assert body["active_users"] == 10
        finally:
            await tc.close()


class TestAnalyticsUserEndpoint:
    async def test_invalid_uuid_returns_400(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        mock_engine = MagicMock()
        app = build_internal_app(redis, engine=mock_engine)  # type: ignore[arg-type]
        server = TestServer(app)
        tc: _TC = TestClient(server)
        await tc.start_server()
        try:
            resp = await tc.get("/internal/analytics/users/not-a-uuid")
            assert resp.status == 400
            body = await resp.json()
            assert "invalid user_id" in body["error"]
        finally:
            await tc.close()

    async def test_no_engine_returns_500(
        self, analytics_client: _TC,
    ) -> None:
        user_id = str(uuid.uuid4())
        resp = await analytics_client.get(
            f"/internal/analytics/users/{user_id}"
        )
        assert resp.status == 500

    async def test_valid_request_returns_200(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        user_id = uuid.uuid4()
        mock_profile = {
            "user_id": str(user_id),
            "total_sessions": 5,
            "total_messages": 20,
            "avg_session_messages": 4.0,
            "avg_session_duration_seconds": 600.0,
            "farewell_rate": 0.4,
            "last_session_at": datetime.now(tz=UTC).isoformat(),
            "dominant_moods": {"venting": 2, "task": 3},
        }
        mock_engine = MagicMock()
        app = build_internal_app(redis, engine=mock_engine)  # type: ignore[arg-type]
        server = TestServer(app)
        tc: _TC = TestClient(server)
        await tc.start_server()
        try:
            with patch(
                "companion_bot_core.internal.routes.get_user_analytics",
                return_value=mock_profile,
            ), patch(
                "companion_bot_core.internal.routes.get_async_session",
            ) as mock_session_ctx:
                mock_ctx = AsyncMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
                mock_session_ctx.return_value = mock_ctx

                resp = await tc.get(
                    f"/internal/analytics/users/{user_id}"
                )
                assert resp.status == 200
                body = await resp.json()
                assert body["total_sessions"] == 5
        finally:
            await tc.close()

    async def test_invalid_days_returns_400(
        self, redis: fakeredis.FakeRedis,
    ) -> None:
        user_id = str(uuid.uuid4())
        mock_engine = MagicMock()
        app = build_internal_app(redis, engine=mock_engine)  # type: ignore[arg-type]
        server = TestServer(app)
        tc: _TC = TestClient(server)
        await tc.start_server()
        try:
            resp = await tc.get(
                f"/internal/analytics/users/{user_id}?days=0"
            )
            assert resp.status == 400
        finally:
            await tc.close()
