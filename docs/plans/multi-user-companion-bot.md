# Plan: Multi-User Configurable Telegram Companion Bot (v1)

## Overview
Production-ready Telegram bot where each user has a personalized, evolvable companion persona driven by per-user prompts.
v1 supports prompt-based skills only (no user-executable code), detects in-chat intent to change tone/behavior, and refines/compacts per-user prompt state asynchronously.
Chat responses use ChatGPT Chat API; prompt-refinement jobs use Codex/Claude in non-interactive mode.
Runtime: `aiogram 3.x`, persistence: `PostgreSQL + Redis`.

## Validation Commands
- `pytest tests/unit/`
- `pytest tests/integration/`
- `pytest tests/security/`
- `pytest tests/data/`
- `ruff check .`
- `mypy .`

### Task 1: Project scaffolding and configuration
- [x] Initialize Python project with `aiogram 3.x`, `asyncpg`, `redis-py`, `alembic`
- [x] Set up environment-based secrets management (no hardcoded tokens)
- [x] Configure structured logging with correlation IDs
- [x] Add `ruff` and `mypy` to dev dependencies
- [x] Mark completed

### Task 2: Database schema and migrations
- [x] Create `users` table (`id`, `telegram_user_id`, `created_at`, `status`, `locale`, `timezone`)
- [x] Create `user_profiles` table (`user_id`, `persona_name`, `tone`, `style_constraints`, `safety_level`, `updated_at`)
- [x] Create `prompt_snapshots` table (`id`, `user_id`, `version`, `system_prompt`, `skill_prompts_json`, `source`, `created_at`)
- [x] Create `conversation_messages` table with `ttl_expires_at` column
- [x] Create `memory_compactions`, `behavior_change_events`, `jobs`, `audit_log` tables
- [x] Write Alembic migration files
- [x] Mark completed

### Task 3: Redis integration
- [x] Configure `refinement_jobs` and `retry_jobs` queue topics
- [x] Implement per-user and global rate limiting
- [x] Implement short-lived prompt context cache
- [x] Implement idempotency keys for Telegram update deduplication
- [x] Mark completed

### Task 4: Telegram ingress service
- [x] Set up `aiogram` webhook/polling receiver
- [x] Implement auth and user routing
- [x] Register `/start`, `/profile`, `/set_tone`, `/set_persona`, `/memory_compact_now`, `/reset_persona`, `/privacy`, `/delete_my_data` command handlers
- [x] Wire update deduplication via Redis idempotency keys
- [x] Mark completed

### Task 5: Chat inference adapter
- [x] Implement `generate_reply(user_context, message) -> reply, usage, safety_flags` interface
- [x] Build ChatGPT Chat API client with retry and exponential backoff
- [x] Add circuit breaker when model provider error rate crosses threshold
- [x] Validate model output schema
- [x] Mark completed

### Task 6: Prompt state manager
- [x] Implement prompt snapshot versioning (immutable snapshots, atomic active pointer)
- [x] Build prompt merge: base system template + user persona segment + skill packs + short-term window + compacted long-term profile
- [x] Implement rollback to previous snapshot on user command or failed quality checks
- [x] Mark completed

### Task 7: Behavior change detector
- [x] Implement intent classifier: `tone_change`, `persona_change`, `skill_add_prompt`, `skill_remove`, `safety_override_attempt`, `normal_chat`
- [x] Implement risk-level classification (low / medium / high) with deterministic heuristics
- [x] Implement confidence thresholding: below threshold defaults to normal chat + clarification question
- [x] Implement auto-apply for low-risk, confirm flow for medium-risk, refuse for high-risk
- [x] Mark completed

### Task 8: Conversation orchestrator
- [x] Build context assembly: user profile + latest prompt snapshot + recent message window
- [x] Wire behavior change detector into chat flow
- [x] Implement confirmation dialogue state for medium-risk config changes
- [x] Persist response metadata and enqueue optional refinement trigger
- [x] Mark completed

### Task 9: Refinement worker
- [x] Implement `refine_prompt(snapshot, recent_context) -> proposed_snapshot_delta, rationale, risk_flags` interface
- [x] Build Codex/Claude non-interactive refinement client
- [x] Implement scheduler: enqueue jobs by cadence and activity thresholds
- [x] Validate refinement output schema and policy; store new versioned snapshot
- [x] Add dead-letter queue for repeated failed jobs
- [x] Emit audit event and optional user-visible "profile updated" notice
- [x] Mark completed

### Task 10: Policy guardrail layer
- [x] Implement prompt-injection checks
- [x] Implement unsafe-role-change checks
- [x] Implement risky capability confirmation flow
- [x] Add per-user rate limits and abuse throttling
- [x] Mark completed

### Task 11: Internal service endpoints
- [x] Implement `POST /internal/refine/{user_id}` to enqueue refinement job
- [x] Implement `POST /internal/detect-change` to classify configuration intent
- [x] Mark completed

### Task 12: Observability
- [x] Add p50/p95 chat latency metrics
- [x] Add detector precision proxy (confirmation reversals) metric
- [x] Add refinement success/failure rate and prompt rollback rate metrics
- [x] Add token usage per user/day/provider metrics
- [x] Add structured event logs with user and request correlation IDs
- [x] Add tracing spans: ingress -> detector -> prompt manager -> model adapter -> persistence
- [x] Mark completed

### Task 13: Privacy and data controls
- [ ] Implement configurable TTL expiration for `conversation_messages`
- [ ] Implement `/delete_my_data` hard-delete flow (preserves audit minimality)
- [ ] Implement PII redaction in logs
- [ ] Encrypt sensitive fields at rest
- [ ] Mark completed

### Task 14: Unit tests
- [ ] Test prompt merge builder correctness
- [ ] Test detector intent mapping and risk policy transitions
- [ ] Test snapshot versioning and rollback atomicity
- [ ] Mark completed

### Task 15: Integration tests
- [ ] End-to-end Telegram update to response for new and existing user
- [ ] Async refinement job updates snapshot without blocking chat
- [ ] Confirmation flow for medium-risk changes
- [ ] Mark completed

### Task 16: Security and data tests
- [ ] Verify prompt-injection attempts do not override policy
- [ ] Verify unsafe capability requests are refused
- [ ] Verify TTL expiration removes eligible conversation rows
- [ ] Verify `/delete_my_data` removes personal records and preserves required audit minimality
- [ ] Mark completed

### Task 17: Load tests
- [ ] Concurrent multi-user chats preserve isolation and latency SLO
- [ ] Mark completed

### Task 18: Phase 0 rollout — local dev
- [ ] Wire fake model adapters and seed prompts for local development
- [ ] Verify all acceptance criteria locally
- [ ] Mark completed
