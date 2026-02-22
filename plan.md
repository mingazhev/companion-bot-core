# RFC: Multi-User Configurable Telegram Companion Bot (Python, v1)

## Summary
Design and ship a production-ready Telegram bot where each user has a personalized, evolvable companion persona driven by per-user prompts.
v1 supports prompt-based skills only (no user-executable code), detects in-chat intent to change tone/behavior, and refines/compacts per-user prompt state asynchronously.
Chat responses use ChatGPT Chat API; prompt-refinement jobs use Codex/Claude in non-interactive mode.

## Decisions Locked
- Runtime framework: `aiogram 3.x`
- Persistence: `PostgreSQL + Redis`
- v1 capability scope: prompt-only skills
- Data policy: minimal retention with TTL + deletion controls
- Prompt compaction/refinement: async background jobs
- Safety model: moderate guardrails

## Goals
- Multi-user isolation of persona, memory profile, and skill prompt packs.
- Natural "on-the-fly" behavior updates from user messages.
- Stable low-latency chat path independent from refinement latency.
- Auditable prompt evolution history with rollback.
- Clear extension path to v2 programmable Python skills.

## Non-Goals (v1)
- Executing user-authored Python code.
- Arbitrary external tool invocation by user prompts.
- Cross-user memory sharing or social graph features.
- Voice/image modalities.

## High-Level Architecture
- Telegram Ingress Service (`aiogram`): webhook/polling receiver, auth, routing.
- Conversation Orchestrator: builds model input from user profile + active prompt state + short memory window.
- Behavior Change Detector: classifies whether incoming message is normal chat vs configuration intent.
- Prompt State Manager: applies approved config changes, versions prompt artifacts.
- Chat Inference Adapter: ChatGPT Chat API client for user-facing replies.
- Refinement Worker: queued periodic compaction/refinement via Codex/Claude non-interactive calls.
- Policy Guardrail Layer: prompt-injection checks, unsafe-role-change checks, risky capability confirmation flow.
- Storage Layer: Postgres for durable state, Redis for queues/rate limiting/cache.
- Observability Layer: structured logs, metrics, tracing, audit events.

## Data Model (PostgreSQL)
- `users`: `id`, `telegram_user_id`, `created_at`, `status`, `locale`, `timezone`.
- `user_profiles`: `user_id`, `persona_name`, `tone`, `style_constraints`, `safety_level`, `updated_at`.
- `prompt_snapshots`: `id`, `user_id`, `version`, `system_prompt`, `skill_prompts_json`, `source` (`manual|detected|refined`), `created_at`.
- `conversation_messages`: `id`, `user_id`, `role`, `content_redacted`, `token_count`, `created_at`, `ttl_expires_at`.
- `memory_compactions`: `id`, `user_id`, `input_window_ref`, `compacted_profile_delta`, `quality_score`, `created_at`.
- `behavior_change_events`: `id`, `user_id`, `message_id`, `detected_intent`, `risk_level`, `action_taken`, `created_at`.
- `jobs`: `id`, `type`, `user_id`, `status`, `attempts`, `scheduled_at`, `started_at`, `finished_at`.
- `audit_log`: `id`, `user_id`, `actor` (`user|system|worker`), `event_type`, `payload_json`, `created_at`.

## Redis Usage
- Queue topics: `refinement_jobs`, `retry_jobs`.
- Rate limits: per user + global.
- Short-lived context cache: last assembled prompt context for fast retries.
- Idempotency keys for Telegram update deduplication.

## Public Interfaces / Contracts
- Telegram commands:
- `/start` initialize user state.
- `/profile` view current persona/tone/skills summary.
- `/set_tone <value>` explicit tone override.
- `/set_persona <text>` explicit persona override.
- `/memory_compact_now` enqueue refinement.
- `/reset_persona` rollback to baseline snapshot.
- `/privacy` show retention and delete options.
- `/delete_my_data` hard-delete user data (subject to audit policy).
- Internal service contract: `POST /internal/refine/{user_id}` enqueue refinement job.
- Internal service contract: `POST /internal/detect-change` classify configuration intent.
- Model adapter interface:
- `generate_reply(user_context, message) -> reply, usage, safety_flags`
- `refine_prompt(snapshot, recent_context) -> proposed_snapshot_delta, rationale, risk_flags`

## Core Flows
- Chat flow:
- Receive Telegram update.
- Load user profile + latest prompt snapshot + recent message window.
- Run behavior-change detection.
- If config intent detected and safe low-risk, update prompt state immediately.
- If medium/high-risk, request explicit confirmation in chat before applying.
- Generate chat response via ChatGPT adapter.
- Persist response metadata and enqueue optional refinement trigger.
- Refinement flow:
- Scheduler enqueues user jobs by cadence and activity thresholds.
- Worker fetches latest snapshot + bounded context.
- Calls Codex/Claude non-interactive refinement prompt.
- Validates output schema and policy.
- Stores new versioned prompt snapshot.
- Emits audit event and optional user-visible “profile updated” notice.

## Prompt Strategy
- Base system prompt template with fixed safety/policy clauses.
- User persona segment (editable by user and detector).
- Skill prompt pack (named prompt modules, prompt-only in v1).
- Short-term memory window (recent N exchanges).
- Compacted long-term profile (facts/preferences/conversation style).
- Versioning rule: every change creates immutable snapshot; active pointer moves atomically.
- Rollback rule: revert active pointer to previous snapshot on user command or failed quality checks.

## Behavior Change Detection Policy
- Intents: `tone_change`, `persona_change`, `skill_add_prompt`, `skill_remove`, `safety_override_attempt`, `normal_chat`.
- Risk levels:
- Low: style/tone tweaks.
- Medium: persona-role changes affecting advice/authority.
- High: policy-bypass or harmful capability requests.
- Actions:
- Low: auto-apply and notify.
- Medium: ask confirm.
- High: refuse and explain constraints.
- Detector implementation: lightweight classifier prompt + deterministic heuristics.
- Confidence thresholding: below threshold defaults to normal chat + clarification question.

## Security, Privacy, Compliance
- Secrets via environment manager (no hardcoded API tokens).
- Encrypt sensitive fields at rest where practical.
- Data minimization default.
- Configurable TTL for `conversation_messages`.
- User-triggered deletion endpoint/command.
- PII redaction in logs.
- Strict model output schema validation for refinement jobs.
- Per-user rate limits and abuse throttling.

## Reliability and Failure Handling
- Telegram update idempotency to avoid duplicate replies.
- Retry policy with exponential backoff for model/API failures.
- Dead-letter queue for repeated failed refinement jobs.
- Circuit breaker when model provider error rate crosses threshold.
- Graceful degradation: continue chat with last valid snapshot if refinement fails.

## Observability
- Metrics:
- p50/p95 chat latency.
- detector precision proxy (confirmation reversals).
- refinement success/failure rate.
- prompt rollback rate.
- token usage per user/day/provider.
- Logs:
- structured event logs with user and request correlation IDs.
- Tracing:
- ingress -> detector -> prompt manager -> model adapter -> persistence spans.

## Testing and Acceptance Scenarios
- Unit tests:
- prompt merge builder correctness.
- detector intent mapping and risk policy transitions.
- snapshot versioning and rollback atomicity.
- Integration tests:
- end-to-end Telegram update to response for new and existing user.
- async refinement job updates snapshot without blocking chat.
- confirmation flow for medium-risk changes.
- Security tests:
- prompt-injection attempts do not override policy.
- unsafe capability requests are refused.
- Data tests:
- TTL expiration removes eligible conversation rows.
- `/delete_my_data` removes personal records and preserves required audit minimality.
- Load tests:
- concurrent multi-user chats preserve isolation and latency SLO.
- Acceptance criteria:
- user can change tone/persona in natural language and via commands.
- user state remains isolated across accounts.
- refinement improves compact profile without changing prohibited guardrails.
- failures do not crash chat path.

## Rollout Plan
- Phase 0: local dev with fake adapters and seed prompts.
- Phase 1: limited beta with telemetry-only detector decisions.
- Phase 2: enable low-risk auto-apply; medium-risk confirmation.
- Phase 3: enable scheduled refinement with conservative cadence.
- Phase 4: tune thresholds and retention defaults from production metrics.

## v2 Extension (Programmable Python Skills)
- Introduce signed skill packages with manifest schema.
- Execute in sandboxed runtime (container or microVM, no host FS/network by default).
- Permission model per skill (explicit user grants).
- Capability broker API instead of direct imports.
- Static checks + runtime quotas (CPU/memory/timeouts).
- Security review gate before enabling user-authored code in production.

## Assumptions and Defaults
- Telegram bot token and provider API keys are available at deploy time.
- Single-region initial deployment is acceptable.
- English-first prompt templates; localization later.
- Moderate guardrails prioritize safety over maximal flexibility.
- If detector confidence is low, bot asks clarification instead of applying config changes.
