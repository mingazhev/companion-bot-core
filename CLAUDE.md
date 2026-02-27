# TdBot — AI Knowledge Base

## Project structure

```
src/tdbot/
  bot/          — aiogram app, handlers (commands + message), outer middleware
  orchestrator/ — full message pipeline: context loader, dialogue state, orchestrator
  behavior/     — intent classifier (tone_change, persona_change, skill_*, safety_override_attempt, normal_chat)
  prompt/       — snapshot versioning (SnapshotRecord), merge builder, rollback
  inference/    — OpenAI Chat API client, circuit breaker, response schemas
  refinement/   — async worker, refinement model client, delta validator
  policy/       — guardrails (prompt injection, unsafe roles), abuse throttle
  privacy/      — TTL sweeper, hard-delete, PII redactor, field encryption
  redis/        — rate limiting, idempotency, job queues, prompt cache
  db/           — SQLAlchemy async models, engine factory
  internal/     — aiohttp internal HTTP service + Prometheus endpoints
  dev/          — FakeChatAPIClient and seed personas for local dev
  metrics.py    — Prometheus counter/histogram registry (singleton)
  tracing.py    — in-process span context managers (structlog-based, NOT OpenTelemetry)
  logging_config.py — structlog configuration with correlation IDs
  signals.py    — shared regex signal scoring (compiled patterns + weight-based scorer)
  config.py     — pydantic-settings Settings singleton; access via get_settings()
  main.py       — application entry point; wires bot, refinement worker, TTL sweeper, internal server
```

## Key architectural patterns

- Settings singleton: `get_settings()` returns a cached `Settings` instance. Tests must patch `tdbot.config._settings` before imports that call it.
- DB session lifecycle: `get_async_session(engine)` is an async context manager that opens a `session.begin()` block and commits on clean exit. The `async_sessionmaker` is cached per engine via `_get_session_factory()`.
- Middleware injection: `IngressMiddleware` injects `db_user`, `db_session`, `tg_user`, and `redis` into the aiogram handler data dict via the middleware pattern. The dispatcher's `workflow_data` additionally provides `snapshot_store`, `chat_client`, `encryptor`, and `settings` to all handlers.
- Fake adapter mode: `USE_FAKE_ADAPTERS=true` replaces `ChatAPIClient` with `FakeChatAPIClient` and uses `InMemorySnapshotStore`. In production mode, `PostgresSnapshotStore` (backed by `prompt_snapshots` table with Redis active-pointer caching) is used. Seed personas from `dev/seeds.py`.
- User provisioning: `get_or_create_user(session, telegram_user_id)` in `bot/users.py` uses a PostgreSQL upsert (`INSERT … ON CONFLICT DO NOTHING` on `telegram_user_id`) followed by a SELECT. This is race-condition-safe for concurrent first messages. The caller (`IngressMiddleware`) is responsible for committing the enclosing transaction.
- Internal route body parsing: always use `await request.read()` to read the request body; do not rely on `Content-Length`. This supports chunked transfer encoding. Parse JSON manually from the raw bytes with `json.loads()`. See `internal/routes.py` for the reference pattern.
- Orchestrator pipeline: `process_message()` runs a strict step sequence: (0a) abuse block check, (0b) guardrail checks (prompt injection, unsafe role, risky capability), (1) pending confirmation dialogue resolution, (2) behavior intent classification, (3) route by action (refuse/confirm/auto_apply/pass_through), (4) context assembly, (5) inference call, (6) message persistence, (7) refinement trigger. `CHAT_LATENCY` is recorded on every exit path including early exits.
- Dual-path profile updates: User profile and prompt snapshots can be modified via two paths: (1) explicit commands (`/set_tone`, `/set_persona`, `/reset_persona`) in `bot/handlers.py`, and (2) in-chat behavior detection in `orchestrator/orchestrator.py` (`_apply_behavior_change`). For `tone_change` and `persona_change`, both paths use `rebuild_and_save_snapshot` from `prompt/helpers.py`. For `skill_add_prompt` and `skill_remove`, the orchestrator uses private helpers (`_add_skill_to_snapshot`, `_remove_skill_from_snapshot`) defined in `orchestrator/orchestrator.py`. When adding new profile fields, ensure both paths are updated.
- Refinement worker queue priority: The worker drains the retry queue (`retry_jobs`, non-blocking 1-second poll) before blocking on the primary queue (`refinement_jobs`, 30-second poll). This ensures retried jobs are processed promptly.
- Refinement worker noop detection: `process_one_job` compares the newly-built snapshot against the current snapshot after applying the delta. If both `system_prompt` and `skill_prompts_json` are identical, the save is skipped and the job is marked `skipped` (no new snapshot row, no user notice, no Redis pointer update).
- Refinement worker circuit-breaker handling: when `CircuitBreakerOpen` is raised during inference, the job is re-enqueued to the primary queue without incrementing `attempt`, using a separate `cb_retries` counter (max 20). This is distinct from regular failures which increment `attempt` and use the retry queue. The 30-second blocking poll provides natural back-off before the circuit-open job is retried.
- `/memory_compact_now` dedup guard: the handler acquires `refinement:pending:{user_uuid}` (SET NX EX 600) before enqueuing. If the guard cannot be set (a job is already in-flight), the request is rejected with a user-facing message. On enqueue failure, the guard is released. This is the same guard used by the orchestrator's `_maybe_enqueue_refinement`, so a manual compaction blocks automatic refinement enqueuing until the job is consumed.
- Snapshot store session parameter: `SnapshotStore` protocol methods (`save`, `get`, `set_active`) accept an optional `session` parameter. When provided, the operation participates in the caller's DB transaction (committed atomically). When `None`, a private session is opened and committed immediately.
- Deferred Redis writes: `PostgresSnapshotStore.set_active()` defers the Redis active-pointer write when a DB session is provided — the write is stored in `session.info["_snapshot_deferred_redis_writes"]`. Callers must call `extract_deferred_redis_writes(session)` while the session is still open to obtain the writes list, then call `flush_deferred_redis_writes(writes, redis)` after the transaction commits. This prevents the Redis pointer from referencing an uncommitted or rolled-back snapshot row, and avoids accessing `session.info` after the session context manager exits. `IngressMiddleware` performs this flush with up to 3 retries (exponential back-off). The refinement worker calls `flush_deferred_redis_writes` directly after its session commits.
- Idempotency key error handling: When a handler raises an exception, `IngressMiddleware` clears the Redis idempotency key (`clear_update_key`) so that Telegram retries of the same `update_id` are not permanently suppressed. After a successful commit, the idempotency key is preserved and deferred Redis writes are flushed.

## Build and test commands

```bash
pytest tests/unit/          # unit tests
pytest tests/integration/   # integration tests
pytest tests/security/      # security tests (prompt injection, unsafe caps)
pytest tests/data/          # TTL expiry and hard-delete flows
pytest tests/load/          # concurrent isolation and latency SLO
ruff check .                # lint
mypy .                      # type-check (strict)
alembic upgrade head        # apply DB migrations
tdbot                       # run the bot
```

## Observability

- Prometheus metrics: `metrics.py` defines all counters and histograms. `GET /metrics` on the internal service exposes them.
- Tracing: `from tdbot.tracing import span, sync_span` — creates structlog-annotated spans propagated via `contextvars`. Not OpenTelemetry.
- Structured logging: always use `log = get_logger(__name__)` from `tdbot.logging_config`, never stdlib `logging` directly.
- `CHAT_LATENCY` is recorded only for requests that reach the model inference call (`auto_apply` and `pass_through` actions) and for all early-exit paths (refuse, confirm, pending-change confirmation/cancellation). Do not skip latency recording on new early-exit paths.

## Redis key layout

- `idempotency:update:{update_id}` — dedup Telegram updates
- `rate_limit:user:{telegram_user_id}` — per-user sliding window (uses Telegram ID, not internal UUID)
- `rate_limit:global` — global RPS cap
- `pending_change:{user_uuid}` — serialised `PendingChange` awaiting confirmation (5-minute TTL)
- `activity_count:{user_uuid}` — message counter triggering refinement at threshold
- `refinement:notice:{user_uuid}` — flag set when refinement completes (cleared on next message)
- `refinement:last_scheduled:{user_uuid}` — timestamp of last cadence-triggered refinement
- `abuse:violations:{user_uuid}` — sorted set of violation timestamps
- `abuse:block:{user_uuid}` — block flag with TTL
- `prompt_cache:{user_uuid}` — cached prompt snapshot (5-minute TTL); not yet wired into context loader
- `refinement:pending:{user_uuid}` — SET-NX guard preventing duplicate refinement enqueues (10-minute TTL)
- `prompt:active:{user_uuid}` — active snapshot pointer (PostgresSnapshotStore)
- `prompt:version:{user_uuid}` — monotonic snapshot version counter (PostgresSnapshotStore)
- `refinement_jobs` / `retry_jobs` — Redis list queues for the worker

## Known incomplete features (explicitly deferred)

- Webhook mode — raises `NotImplementedError`; only polling is functional
- Prompt cache (`prompt_cache:{user_uuid}`) is defined in `redis/prompt_cache.py` but not yet wired into the context loader pipeline

## Security notes

- `FIELD_ENCRYPTION_KEY` must be set when `ENCRYPT_SENSITIVE_FIELDS=true`; absence raises `RuntimeError` at startup. Field encryption is wired into conversation message content and user profile fields (persona_name, tone). Legacy unencrypted rows are handled gracefully via `decrypt_safe`.
- Internal HTTP service must NOT bind to `0.0.0.0`; there is no authentication on internal routes. A `Settings` validator rejects `0.0.0.0` and `::` at startup.
- Rate limit pipeline uses Redis non-transactional batching (not MULTI/EXEC); allows ~1 extra request through under concurrent load
