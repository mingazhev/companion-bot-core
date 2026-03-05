# Bot Improvements Plan

## Priority 1: Dialog Quality

### 1.1 Emotion Detector

**Goal:** Classify user message emotion before inference, inject mode-specific instructions into system prompt.

**Files:**
- New: `src/companion_bot_core/behavior/emotion.py`
- Edit: `src/companion_bot_core/orchestrator/orchestrator.py` — add step between context assembly and inference
- ~~Edit: `src/companion_bot_core/prompt/merge_builder.py`~~ — not needed; injection done via `model_copy()` in orchestrator
- New: `tests/unit/test_emotion_detector.py`

**Implementation:**
- [x] Create `EmotionMode` enum: `venting`, `validation`, `task`, `farewell`, `neutral`
- [x] Build regex-based classifier using signal patterns (similar to `signals.py`)
- [x] Create mode-to-instruction map
- [x] Inject instruction as last paragraph of system prompt before inference call
- [x] Add metric: `emotion_detected_total` (labels: mode)

**Tests:**
- [x] Unit: classifier returns correct mode for 10+ example messages per mode
- [x] Unit: instruction mapping includes/excludes injection correctly
- [x] Unit: edge cases (empty strings, case insensitivity, mixed signals)

**Acceptance:** Round 6 test scenarios (Katya venting, Artem self-doubt) pass without advice/menu in response.

---

### 1.2 Repetition Guard

**Goal:** Post-process bot response before sending — detect and eliminate phrase repetition from recent history.

**Files:**
- New: `src/companion_bot_core/orchestrator/response_filter.py`
- Edit: `src/companion_bot_core/orchestrator/orchestrator.py` — add post-inference step
- New: `tests/unit/test_response_filter.py`

**Implementation:**
- [x] Extract key phrases from new response (sentences or clauses)
- [x] Compare against last 3-5 bot messages using n-gram overlap (trigrams, threshold ~0.6)
- [x] If overlap detected: Option B (strip repeated sentence) with fallback to Option A (re-call with anti-repetition instruction)
- [x] Max 1 re-call to avoid latency spiral
- [x] Add metric: `repetition_guard_triggered_total`

**Dependencies:** None, can be built independently.

**Tests:**
- [x] Unit: overlap detection correctly identifies repeated phrases
- [x] Unit: stripping preserves coherent response
- [x] Unit: edge cases — short responses, single sentence, greeting-only

**Acceptance:** Bot does not repeat "нормально/сбежать" when user switches topic (Round 6.5 Katya msg 4 scenario).

---

### 1.3 Topic Tracker

**Goal:** Track current conversation topic, inject context on topic switches.

**Files:**
- New: `src/companion_bot_core/orchestrator/topic_tracker.py`
- Edit: `src/companion_bot_core/orchestrator/orchestrator.py`
- New: `tests/unit/test_topic_tracker.py`

**Implementation:**
- [x] Topic detection via keyword extraction from user messages (simple keyword matching)
- [x] Store current topic in Redis: `topic:{user_uuid}` (TTL: session duration, ~30 min)
- [x] Detect topic switch signals: "кстати", "а вот ещё", "забей", "ладно, другое", "сменим тему"
- [x] On switch: inject "Пользователь перешёл к новой теме. Не возвращайся к предыдущей." into system prompt
- [x] Store previous topic briefly for "warm return" feature (2b)
- [x] Add metric: `topic_switch_total`

**Dependencies:** 1.1 (shares injection mechanism into system prompt).

**Tests:**
- [x] Unit: topic switch detection on signal phrases
- [x] Unit: keyword extraction and divergence detection
- [x] Unit: Redis state management (store/retrieve/previous topic)
- [x] Unit: edge cases (empty strings, case insensitivity, mixed signals)

**Acceptance:** Bot stops prefacing sериалы recommendations with "это нормально хотеть сбежать" (Round 6.5 Katya msg 4).

---

## Priority 2: New Features

### 2.1 Mood Journal

**Goal:** Automatically track user mood across conversations, expose via `/mood` command.

**Files:**
- New: `src/companion_bot_core/db/models.py` — add `MoodEntry` model
- New: `alembic/versions/xxx_add_mood_entries.py`
- Edit: `src/companion_bot_core/orchestrator/orchestrator.py` — save mood after emotion detection
- Edit: `src/companion_bot_core/bot/handlers.py` — add `/mood` command
- New: `tests/unit/test_mood_journal.py`

**Implementation:**
1. `mood_entries` table: id, user_id, mood (enum: happy, sad, anxious, angry, neutral, excited), intensity (1-5), context_snippet (first 50 chars of message), created_at
2. After emotion detector runs (1.1), save mood entry if mode != neutral
3. `/mood` command: show last 7 days as text timeline
4. `/mood week` / `/mood month` for different ranges
5. No diagnosis, no advice based on mood patterns (Phase 1)

**Dependencies:** 1.1 (emotion detector provides mood classification).

**Tests:**
- Unit: mood saving logic
- Unit: `/mood` output formatting
- Integration: full flow message → mood saved → `/mood` displays it

---

### 2.2 Proactive Messages

**Goal:** Bot initiates contact for warm returns and optional daily check-ins.

**Files:**
- New: `src/companion_bot_core/proactive/scheduler.py`
- New: `src/companion_bot_core/proactive/warm_return.py`
- New: `src/companion_bot_core/proactive/checkin.py`
- Edit: `src/companion_bot_core/db/models.py` — add `user_preferences` fields
- New: `alembic/versions/xxx_add_user_preferences.py`
- Edit: `src/companion_bot_core/bot/handlers.py` — add `/checkin` command
- Edit: `src/companion_bot_core/main.py` — wire proactive scheduler
- New: `tests/unit/test_proactive.py`

**Implementation:**
1. Add to `user_profiles`: `proactive_enabled` (bool, default false), `checkin_time` (time, nullable), `quiet_hours_start`/`quiet_hours_end`
2. Warm return (always on): if last message > 48h ago, on next user message inject context "Давно не общались. Вспомни последнюю тему и отреагируй."
3. Daily check-in (opt-in via `/checkin on 09:00`): scheduled Redis job sends short message
4. Scheduler: background asyncio task polling Redis sorted set of scheduled events
5. Respect quiet hours, timezone from user profile
6. `/checkin off` — disable

**Dependencies:** 2.1 (mood journal for check-in context).

**Tests:**
- Unit: scheduler timing logic
- Unit: warm return detection (last message age)
- Unit: quiet hours filtering
- Integration: check-in delivery

---

### 2.3 Goals & Habits

**Goal:** Lightweight habit tracker embedded in dialogue.

**Files:**
- New: `src/companion_bot_core/db/models.py` — add `Habit` model
- New: `alembic/versions/xxx_add_habits.py`
- Edit: `src/companion_bot_core/behavior/detector.py` — add `habit_create` and `habit_check` intents
- Edit: `src/companion_bot_core/orchestrator/orchestrator.py` — handle habit intents
- Edit: `src/companion_bot_core/bot/handlers.py` — add `/habits` command
- New: `tests/unit/test_habits.py`

**Implementation:**
1. `habits` table: id, user_id, title, frequency (daily/weekly), current_streak, best_streak, last_checked_at, created_at, archived_at
2. New intents in detector: "хочу привычку", "хочу каждый день X" → `habit_create`
3. Check-in via natural language: "сегодня читала" → `habit_check` (match against existing habits)
4. `/habits` — list active habits with streaks
5. Proactive reminder (if 2.2 enabled): "Кстати, сегодня читала?" — only once per day per habit
6. No gamification, no guilt. Missed → streak resets silently

**Dependencies:** 2.2 (proactive scheduler for reminders).

**Tests:**
- Unit: streak calculation
- Unit: intent detection for habit create/check
- Unit: `/habits` formatting

---

### 2.4 Conversation Bookmarks

**Goal:** User can save important moments from conversation.

**Files:**
- New: `src/companion_bot_core/db/models.py` — add `Bookmark` model
- New: `alembic/versions/xxx_add_bookmarks.py`
- Edit: `src/companion_bot_core/behavior/detector.py` — add `bookmark_request` intent
- Edit: `src/companion_bot_core/orchestrator/orchestrator.py` — handle bookmark intent
- Edit: `src/companion_bot_core/bot/handlers.py` — add `/bookmarks` command
- New: `tests/unit/test_bookmarks.py`

**Implementation:**
1. `bookmarks` table: id, user_id, user_message (text), bot_response (text), tag (optional), created_at
2. Intent triggers: "запомни это", "это важно", "сохрани"
3. On trigger: save previous user message + bot response pair
4. `/bookmarks` — list saved moments (last 20, with dates)
5. `/bookmarks search <query>` — simple text search
6. Context injection: refinement worker can reference bookmarks in long_term_profile

**Dependencies:** None.

**Tests:**
- Unit: intent detection
- Unit: bookmark save/retrieve
- Unit: `/bookmarks` formatting and search

---

## Priority 3: Metrics & Analytics

### 3.1 Conversation Quality Metrics

**Goal:** Add Prometheus metrics that measure dialog quality, not just infrastructure.

**Files:**
- Edit: `src/companion_bot_core/metrics.py` — add new counters/histograms
- Edit: `src/companion_bot_core/orchestrator/orchestrator.py` — emit metrics

**Implementation:**
1. `response_length_sentences` histogram (buckets: 1, 2, 3, 5, 7, 10, 15)
2. `emotion_detected_total` counter (labels: mode) — from 1.1
3. `repetition_guard_triggered_total` counter — from 1.2
4. `topic_switch_total` counter — from 1.3
5. `session_messages_total` histogram (buckets: 1, 3, 5, 7, 10, 15, 20)
6. `farewell_detected_total` counter

**Dependencies:** 1.1, 1.2, 1.3 (metrics come from those components).

---

### 3.2 Session Analytics

**Goal:** Group messages into sessions, track engagement over time.

**Files:**
- New: `src/companion_bot_core/db/models.py` — add `ConversationSession` model
- New: `alembic/versions/xxx_add_sessions.py`
- Edit: `src/companion_bot_core/orchestrator/orchestrator.py` — session boundary detection
- New: `src/companion_bot_core/internal/analytics.py`
- Edit: `src/companion_bot_core/internal/routes.py` — add analytics endpoints
- New: `tests/unit/test_sessions.py`

**Implementation:**
1. `conversation_sessions` table: id, user_id, started_at, ended_at, message_count, dominant_mood, ended_with_farewell (bool)
2. Session boundary: gap > 30 min between messages = new session
3. Update session on each message (message_count++, ended_at = now)
4. Close session when farewell detected or timeout
5. `GET /internal/analytics/overview` — active users, avg session length, return rate
6. `GET /internal/analytics/users/{user_id}` — per-user engagement profile

**Dependencies:** 1.1 (mood for dominant_mood field).

---

### 3.3 In-Bot Feedback

**Goal:** Collect user satisfaction naturally within conversation.

**Files:**
- New: `src/companion_bot_core/db/models.py` — add `FeedbackEntry` model
- New: `alembic/versions/xxx_add_feedback.py`
- New: `src/companion_bot_core/orchestrator/feedback.py`
- Edit: `src/companion_bot_core/orchestrator/orchestrator.py` — trigger feedback ask
- New: `tests/unit/test_feedback.py`

**Implementation:**
1. `feedback_entries` table: id, user_id, session_id, raw_text, sentiment_score (1-5), created_at
2. Trigger: every 10th session (configurable), at natural pause point (after farewell)
3. Bot asks: "Кстати, как тебе наше общение? Можешь одним словом или оценкой"
4. Next user response classified by model into 1-5 sentiment
5. Metric: `user_feedback_score` histogram
6. Never ask more than once per week per user

**Dependencies:** 3.2 (session tracking for trigger cadence).

---

## Priority 4: Testing

### 4.1 Persona Test Scenarios (local, via subagent)

**Goal:** YAML-defined test scenarios runnable through Claude Code subagent.

**Files:**
- New: `tests/persona/scenarios/katya_empathy.yaml`
- New: `tests/persona/scenarios/katya_recommendations.yaml`
- New: `tests/persona/scenarios/artem_selfdoubt.yaml`
- New: `tests/persona/scenarios/artem_technical.yaml`
- New: `tests/persona/runner.py` — scenario executor using chat.py
- New: `tests/persona/judge.py` — LLM-as-judge evaluator
- New: `tests/persona/checks.py` — deterministic checks (shared with 4.2)

**Implementation:**
1. YAML schema: persona metadata + message list + per-message checks
2. Runner: reset bot data → onboard persona → send messages → collect responses → apply checks
3. Judge: send full dialogue to LLM with scoring prompt → parse JSON scores
4. Output: markdown report to `docs/reports/auto/`
5. Invoked via: "Run persona test katya_empathy" in Claude Code

**Dependencies:** None (uses existing chat.py infrastructure).

---

### 4.2 Deterministic Quality Checks (CI)

**Goal:** Fast unit tests that validate response quality patterns without LLM.

**Files:**
- New: `src/companion_bot_core/quality/checks.py` — reusable check functions
- New: `tests/unit/test_response_quality.py`
- New: `tests/unit/fixtures/response_pairs.json` — saved (input, response) pairs from persona tests

**Implementation:**
1. Check functions:
   - `has_ai_markers(text) -> list[str]` — returns found markers
   - `count_bullet_points(text) -> int`
   - `count_sentences(text) -> int`
   - `has_menu_pattern(text) -> bool` — detects "1) ... 2) ..." or "- X\n- Y\n- Z" with 3+ items
   - `is_short_farewell(text, max_sentences=3) -> bool`
   - `contains_name(text, name) -> bool`
   - `ngram_overlap(text_a, text_b, n=3) -> float` — reused by 1.2
2. Fixtures: collect (input, response) pairs from persona tests, commit as JSON
3. Parametrized pytest: each fixture pair validated against relevant checks
4. CI: runs on every PR, fast (no network, no LLM)

**Dependencies:** None.

---

## Priority 5: Model A/B Testing

### 5.1 Model Switch Infrastructure

**Goal:** Support A/B testing between models without code changes.

**Files:**
- Edit: `src/companion_bot_core/config.py` — add experimental model fields
- Edit: `src/companion_bot_core/inference/adapter.py` — model selection logic
- Edit: `src/companion_bot_core/metrics.py` — add model assignment metric
- New: `tests/unit/test_model_ab.py`

**Implementation:**
1. New config fields: `chat_model_experimental` (str), `chat_model_ab_ratio` (float 0.0-1.0)
2. Assignment: hash(user_uuid) % 100 < ratio*100 → experimental model. Deterministic per user.
3. Store assignment in Redis: `model_ab:{user_uuid}` — so user stays on same model across sessions
4. Metric: `model_ab_assignment_total` (labels: model)
5. All existing quality metrics get `model` label for comparison

**Dependencies:** 3.1 (quality metrics for comparison).

**First experiment:** `gpt-5.3-instant` vs current `gpt-5-mini`. Run for 1 week, compare session length + feedback scores.

---

## Execution Order

```
Phase 1 (dialog quality):  1.1 → 1.2 → 1.3 → test round 7
Phase 2 (testing):          4.2 → 4.1 (parallel with Phase 1)
Phase 3 (metrics):          3.1 → 3.2 → 3.3
Phase 4 (features):         2.4 → 2.1 → 2.2 → 2.3
Phase 5 (model):            5.1 → experiment
```

Phases 1 and 2 can run in parallel. Phase 3 depends on Phase 1 components. Phase 4 depends on Phase 1 (emotion detector). Phase 5 depends on Phase 3 (metrics for comparison).
