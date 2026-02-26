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
  config.py     — pydantic-settings Settings singleton; access via get_settings()
```

## Key architectural patterns

- Settings singleton: `get_settings()` returns a cached `Settings` instance. Tests must patch `tdbot.config._settings` before imports that call it.
- DB session lifecycle: `get_async_session(engine)` is an async context manager that opens a `session.begin()` block and commits on clean exit. The `async_sessionmaker` is cached per engine via `_get_session_factory()`.
- Middleware injection: `IngressMiddleware` injects `db_user`, `db_session`, `tg_user`, and `redis` into the aiogram handler data dict via the middleware pattern. The dispatcher's `workflow_data` additionally provides `snapshot_store`, `chat_client`, and `settings` to all handlers.
- Fake adapter mode: `USE_FAKE_ADAPTERS=true` replaces `ChatAPIClient` with `FakeChatAPIClient`. `InMemorySnapshotStore` is always used. Seed personas from `dev/seeds.py`.
- User provisioning: `get_or_create_user(session, telegram_user_id)` in `bot/users.py` uses a PostgreSQL upsert (`INSERT … ON CONFLICT DO NOTHING` on `telegram_user_id`) followed by a SELECT. This is race-condition-safe for concurrent first messages. The caller (`IngressMiddleware`) is responsible for committing the enclosing transaction.
- Internal route body parsing: always use `await request.read()` to read the request body; do not rely on `Content-Length`. This supports chunked transfer encoding. Parse JSON manually from the raw bytes with `json.loads()`. See `internal/routes.py` for the reference pattern.

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
- `refinement_jobs` / `retry_jobs` — Redis list queues for the worker

## Known incomplete features (explicitly deferred)

- `/set_tone`, `/set_persona`, `/reset_persona` — handlers respond but do not persist to DB (deferred to prompt state manager task)
- `/memory_compact_now` — handler acknowledges the command but does not enqueue a job into `refinement_jobs`. The refinement worker is running and processes jobs queued via the activity threshold or `/internal/refine`; the gap is only in the command handler path.
- Webhook mode — raises `NotImplementedError`; only polling is functional
- Field-level encryption — `FieldEncryptor` is implemented but not wired to any DB read/write path
- Policy guardrails — `policy/guardrails.py` and `policy/abuse_throttle.py` exist but are not called from the orchestrator
- Behavior change application — detected changes are recorded in `BehaviorChangeEvent` but do not update the prompt snapshot
- `InMemorySnapshotStore` is used in all modes; a DB-backed store is needed for production persistence
- `refinement/scheduler.py` — cadence-based scheduling exists but is not called from any production path

## Security notes

- `FIELD_ENCRYPTION_KEY` must be set when `ENCRYPT_SENSITIVE_FIELDS=true`; absence raises `RuntimeError` when `FieldEncryptor.from_settings()` is called. Currently `FieldEncryptor` is not constructed at startup (field encryption is not yet wired), so this check is not exercised during normal operation.
- Internal HTTP service must NOT bind to `0.0.0.0`; there is no authentication on internal routes. A `Settings` validator rejects `0.0.0.0` and `::` at startup.
- Rate limit pipeline uses Redis non-transactional batching (not MULTI/EXEC); allows ~1 extra request through under concurrent load
