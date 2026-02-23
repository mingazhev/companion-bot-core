"""Load tests: concurrent multi-user chat isolation and latency SLO.

Contract:
- N users processing messages concurrently do not cross-contaminate each
  other's persona snapshots, conversation history, or pending state.
- p95 pipeline latency under concurrent load stays below the SLO threshold.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis as fakeredis
import pytest

from tdbot.inference.schemas import OpenAIResponse
from tdbot.orchestrator.dialogue_state import get_pending_change
from tdbot.orchestrator.orchestrator import process_message
from tdbot.prompt.schemas import SnapshotRecord
from tdbot.prompt.snapshot_store import InMemorySnapshotStore
from tdbot.redis.queues import QUEUE_REFINEMENT_JOBS, get_queue_length

# SLO: 95th-percentile latency must stay below this value (seconds).
# The pipeline uses mocked HTTP and in-memory data structures; 200 ms is generous.
_LATENCY_SLO_P95_SECONDS = 0.200

# Number of concurrent users for isolation and latency tests.
_CONCURRENT_USERS = 30


# ---------------------------------------------------------------------------
# Shared helpers (mirrors integration test conventions)
# ---------------------------------------------------------------------------


def _make_openai_response(content: str = "Hello!") -> OpenAIResponse:
    """Build a valid OpenAI Chat Completions response."""
    return OpenAIResponse.model_validate(
        {
            "id": "chatcmpl-load-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
            },
        }
    )


def _make_mock_client(reply_text: str) -> AsyncMock:
    """Return a mock ChatAPIClient that always replies with *reply_text*."""
    client = AsyncMock()
    client.chat_completion = AsyncMock(return_value=_make_openai_response(reply_text))
    return client


def _make_db_session(history_rows: list[Any] | None = None) -> AsyncMock:
    """Return a mock async SQLAlchemy session."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = history_rows or []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# Test 1: Each concurrent user receives the reply from their own mock client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_users_each_get_own_reply() -> None:
    """N users chat concurrently; each receives the reply from their own client.

    Verifies that user_id routing is correct under concurrent load: the reply
    returned to user A is the reply configured for user A, not user B's.
    """
    shared_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_store = InMemorySnapshotStore()

    # Each user gets a unique expected reply keyed by their UUID.
    expected: dict[uuid.UUID, str] = {}
    clients: dict[uuid.UUID, AsyncMock] = {}
    for _ in range(_CONCURRENT_USERS):
        uid = uuid.uuid4()
        reply = f"Reply for user {uid}"
        expected[uid] = reply
        clients[uid] = _make_mock_client(reply)

    async def run_one(uid: uuid.UUID) -> tuple[uuid.UUID, str]:
        reply = await process_message(
            user_id=uid,
            message_text="Hello there!",
            session=_make_db_session(),
            snapshot_store=shared_store,
            redis=shared_redis,
            chat_client=clients[uid],
        )
        return uid, reply

    results = await asyncio.gather(*[run_one(uid) for uid in expected])

    for uid, actual_reply in results:
        assert actual_reply == expected[uid], (
            f"User {uid} got wrong reply: expected {expected[uid]!r}, got {actual_reply!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: Persona snapshots are not contaminated by concurrent processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_users_persona_snapshots_not_contaminated() -> None:
    """N users with distinct active snapshots: after concurrent processing each
    user's active snapshot remains their own (no cross-user swap or overwrite).
    """
    shared_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_store = InMemorySnapshotStore()

    user_personas: dict[uuid.UUID, str] = {}
    for _ in range(_CONCURRENT_USERS):
        uid = uuid.uuid4()
        persona = f"You are a unique assistant configured for user {uid}."
        snapshot = SnapshotRecord(
            user_id=uid,
            version=1,
            system_prompt=persona,
            source="initial",
        )
        await shared_store.save(snapshot)
        await shared_store.set_active(uid, snapshot.id)
        user_personas[uid] = persona

    async def run_one(uid: uuid.UUID) -> None:
        await process_message(
            user_id=uid,
            message_text="What can you do for me?",
            session=_make_db_session(),
            snapshot_store=shared_store,
            redis=shared_redis,
            chat_client=_make_mock_client("I can help!"),
        )

    await asyncio.gather(*[run_one(uid) for uid in user_personas])

    # Each user's active snapshot must still be the one set before the load run.
    for uid, original_persona in user_personas.items():
        active = await shared_store.get_active(uid)
        assert active is not None, f"User {uid} lost their active snapshot"
        assert active.system_prompt == original_persona, (
            f"User {uid} persona contaminated: "
            f"expected {original_persona!r}, got {active.system_prompt!r}"
        )


# ---------------------------------------------------------------------------
# Test 3: Pending confirmation state is isolated between concurrent users
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_pending_state_isolation() -> None:
    """Users with pending confirmation states do not interfere with each other.

    Two groups run concurrently:
    - Group A: trigger a medium-risk persona change → pending state written.
    - Group B: send normal messages → no pending state expected.

    After concurrent execution the groups are verified against expectations.
    """
    shared_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_store = InMemorySnapshotStore()

    group_a_users: list[uuid.UUID] = [uuid.uuid4() for _ in range(10)]
    group_b_users: list[uuid.UUID] = [uuid.uuid4() for _ in range(10)]

    async def trigger_persona_change(uid: uuid.UUID) -> None:
        await process_message(
            user_id=uid,
            message_text="From now on you are Max, my personal advisor.",
            session=_make_db_session(),
            snapshot_store=shared_store,
            redis=shared_redis,
            # MagicMock: chat client is not reached for confirm-routed messages.
            chat_client=MagicMock(),
        )

    async def send_normal_message(uid: uuid.UUID) -> str:
        return await process_message(
            user_id=uid,
            message_text="What is the weather like today?",
            session=_make_db_session(),
            snapshot_store=shared_store,
            redis=shared_redis,
            chat_client=_make_mock_client("I don't know the weather."),
        )

    tasks = [trigger_persona_change(uid) for uid in group_a_users] + [
        send_normal_message(uid) for uid in group_b_users
    ]
    await asyncio.gather(*tasks)

    # Group A — every user should have a pending persona_change in Redis.
    for uid in group_a_users:
        pending = await get_pending_change(shared_redis, str(uid))
        assert pending is not None, (
            f"Group A user {uid} lost their pending confirmation state"
        )
        assert pending.detection_result.intent == "persona_change"

    # Group B — no user should have any pending state.
    for uid in group_b_users:
        pending = await get_pending_change(shared_redis, str(uid))
        assert pending is None, (
            f"Group B user {uid} unexpectedly has a pending confirmation state"
        )


# ---------------------------------------------------------------------------
# Test 4: Per-user activity counters stay isolated under concurrent load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_activity_counters_are_isolated() -> None:
    """Each user's activity counter increments independently under concurrency.

    Sends exactly *threshold* messages for each user; exactly one refinement
    job per user should appear in the queue — overflow from another user's
    counter must not trigger an extra job for a different user.
    """
    shared_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_store = InMemorySnapshotStore()

    n_users = 10
    threshold = 3
    users = [uuid.uuid4() for _ in range(n_users)]

    async def send_messages(uid: uuid.UUID, count: int) -> None:
        for i in range(count):
            await process_message(
                user_id=uid,
                message_text=f"Message {i}",
                session=_make_db_session(),
                snapshot_store=shared_store,
                redis=shared_redis,
                chat_client=_make_mock_client("Reply!"),
                refinement_activity_threshold=threshold,
            )

    # All users send exactly *threshold* messages concurrently.
    await asyncio.gather(*[send_messages(uid, threshold) for uid in users])

    # One refinement job should have been enqueued per user.
    total_jobs = await get_queue_length(shared_redis, QUEUE_REFINEMENT_JOBS)
    assert total_jobs == n_users, (
        f"Expected {n_users} refinement jobs (one per user), got {total_jobs}"
    )


# ---------------------------------------------------------------------------
# Test 5: p95 latency SLO under concurrent multi-user load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_chat_latency_slo() -> None:
    """p95 end-to-end latency stays below _LATENCY_SLO_P95_SECONDS under load.

    Runs _CONCURRENT_USERS concurrent message pipelines and computes the 95th
    percentile of wall-clock times.  All infrastructure is mocked so the
    measurement captures algorithmic overhead, not network I/O.
    """
    shared_redis = fakeredis.FakeRedis(decode_responses=True)
    shared_store = InMemorySnapshotStore()
    users = [uuid.uuid4() for _ in range(_CONCURRENT_USERS)]
    latencies: list[float] = []

    async def timed_process(uid: uuid.UUID) -> None:
        t0 = time.perf_counter()
        await process_message(
            user_id=uid,
            message_text="Quick question about the weather.",
            session=_make_db_session(),
            snapshot_store=shared_store,
            redis=shared_redis,
            chat_client=_make_mock_client("It's sunny!"),
        )
        latencies.append(time.perf_counter() - t0)

    await asyncio.gather(*[timed_process(uid) for uid in users])

    assert len(latencies) == _CONCURRENT_USERS

    # statistics.quantiles(data, n=100) returns 99 cut-points (percentiles 1–99).
    # Index 94 (0-based) is the 95th percentile.
    p95 = statistics.quantiles(latencies, n=100)[94]
    assert p95 < _LATENCY_SLO_P95_SECONDS, (
        f"p95 latency {p95 * 1000:.1f} ms exceeds SLO of "
        f"{_LATENCY_SLO_P95_SECONDS * 1000:.0f} ms"
    )
