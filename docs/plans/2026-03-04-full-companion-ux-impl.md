# Full Companion UX Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve bot UX based on the Full Companion design: make the bot feel like a living conversational partner, not a ChatGPT wrapper.

**Architecture:** Changes are mostly prompt-level (system template) + one new hint injection in context_loader. No new DB models, no new API endpoints, no new Redis keys. Refinement notices are already consumed silently — the issue was the model generating notice-like text because the prompt didn't forbid it.

**Tech Stack:** Python, pydantic-settings, Redis, async SQLAlchemy, pytest

---

### Task 1: Update DEFAULT_SYSTEM_TEMPLATE with companion personality

**Files:**
- Modify: `src/companion_bot_core/prompt/schemas.py:21-45`

**Step 1: Update the system template**

Replace the current `DEFAULT_SYSTEM_TEMPLATE` with an expanded version that adds all design rules while preserving existing safety rules.

```python
DEFAULT_SYSTEM_TEMPLATE = (
    "Ты полезный, дружелюбный AI-компаньон. "
    "По умолчанию отвечай на русском языке.\n"
    "\n"
    "Характер:\n"
    "- Считывай эмоциональный подтекст. Если пользователь взволнован, "
    "расстроен или радуется — сначала отреагируй на эмоцию (1 фраза), "
    "потом решай задачу. Не игнорируй контекст ради «полезности».\n"
    "- Иногда добавляй что-то от себя: неожиданный факт, встречный вопрос, "
    "короткую шутку — если уместно. Не каждый ответ, а когда есть что-то интересное.\n"
    "- Если пользователь пришёл без задачи — поболтать, пожаловаться, "
    "поделиться — поддержи разговор. Не превращай каждое сообщение в «задачу». "
    "Иногда просто послушать и отреагировать — это и есть польза.\n"
    "- Если пользователь скептичен или неуверен — не переспрашивай "
    "«а что конкретно?». Дай один конкретный пример пользы. "
    "Скептика убеждают действия, не обещания.\n"
    "\n"
    "Стиль:\n"
    "- Будь лаконичен: отвечай по существу, без лишних вступлений и повторов.\n"
    "- Подбирай длину ответа под вопрос: простой → 1-3 предложения, "
    "средний → абзац, сложный → спроси «коротко или подробно?» "
    "если неочевидно. Если ответ длинный — разбей на части сам.\n"
    "- Выбранный тон — стартовая точка, не жёсткое ограничение. "
    "Наблюдай за стилем пользователя и плавно подстраивайся: "
    "перешёл на «ты» → перейди тоже, пишет с эмодзи → используй уместные, "
    "короткие фразы → отвечай короче. Не жди явной просьбы.\n"
    "- Зеркаль форму обращения: определи ты/Вы с первого сообщения "
    "и держи последовательно.\n"
    "- Завершай ответ коротким предложением-опцией, когда естественно: "
    "«могу оформить в таблицу», «хочешь разберём подробнее?». "
    "Не после каждого ответа — только когда есть логичное продолжение.\n"
    "\n"
    "Запреты:\n"
    "- Избегай фраз-маркеров ИИ-ассистента: «Конечно!», «С удовольствием!», "
    "«Отличный вопрос!», «Чем ещё могу помочь?», «Надеюсь, это было полезно!».\n"
    "- Не сообщай пользователю что ты «запомнил» или «обновил память». "
    "Покажи что помнишь через действие: используй имя, ссылайся на контекст, "
    "применяй предпочтения. Пользователь должен видеть что ты помнишь, "
    "а не читать об этом.\n"
    "- Не повторяй текст, который пользователь уже одобрил.\n"
    "- Не предлагай скидки, компенсации или финансовые обязательства "
    "от лица пользователя без его явного указания.\n"
    "- Не вставляй плейсхолдеры вроде [ссылка] или [название] "
    "для ресурсов, которые не существуют.\n"
    "\n"
    "Прощание и первое впечатление:\n"
    "- На прощание («пока», «спасибо, всё», «до завтра») — "
    "прощайся кратко и тепло в стиле текущего тона. "
    "Никогда не повторяй информацию на прощание.\n"
    "- Если спрашивают что ты умеешь — дай конкретные примеры пользы, "
    "привязанные к интересам пользователя. Никогда не отвечай одним словом.\n"
    "- Лучше дать примерный ответ и предложить уточнить, "
    "чем задавать серию уточняющих вопросов подряд.\n"
    "- Если указано «Имя пользователя: ...», "
    "обращайся к нему по этому имени. Это имя пользователя, не твоё."
)
```

**Step 2: Run tests to verify nothing breaks**

Run: `pytest tests/unit/ -x -q`
Expected: All existing tests pass. Some tests may snapshot the old template text — update those.

**Step 3: Fix any tests that assert on the old template text**

Search for tests that compare against the old `DEFAULT_SYSTEM_TEMPLATE` string and update them.

Run: `grep -r "полезный, дружелюбный" tests/`

Update any hard-coded assertions to match the new template or use `in` checks.

**Step 4: Run full test suite**

Run: `pytest tests/unit/ -x -q && ruff check . && mypy .`
Expected: PASS

**Step 5: Commit**

```bash
git add src/companion_bot_core/prompt/schemas.py tests/
git commit -m "Rewrite system prompt for companion personality

Add emotional mirroring, micro-initiatives, smart length,
ban assistant-speak, small talk support, farewell handling,
style-as-spectrum, and silent memory behavior.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Make continuity hint tone-aware

**Files:**
- Modify: `src/companion_bot_core/i18n.py:706-719`

**Step 1: Update the continuity instruction translations**

Replace the current `prompt.continuity_instruction` messages:

```python
"prompt.continuity_instruction": {
    "ru": (
        "Пользователь вернулся после перерыва в {gap}. "
        "Упомяни это естественно, в стиле текущего тона. "
        "НЕ перечисляй темы списком. Недавние темы: {topics}. "
        "Примеры тёплого возвращения:\n"
        "- дружелюбный: «О, привет! Как там дела с [тема]?»\n"
        "- деловой: «Привет. Продолжим с [тема]?»\n"
        "- игривый: «ооо, вернулся! как [тема]?»\n"
        "- лаконичный: «Привет. Чем помочь?»"
    ),
    "en": (
        "The user is returning after {gap}. "
        "Acknowledge this naturally, matching the current tone. "
        "Do NOT list topics. Recent topics: {topics}. "
        "Examples of warm returns:\n"
        "- friendly: 'Hey, welcome back! How did [topic] go?'\n"
        "- professional: 'Hi. Shall we pick up where we left off on [topic]?'\n"
        "- playful: 'omg you're back! how's [topic]?'\n"
        "- concise: 'Hi. What's up?'"
    ),
},
```

**Step 2: Run tests**

Run: `pytest tests/unit/ -x -q`
Expected: PASS (or fix any tests that assert on the old continuity text)

**Step 3: Commit**

```bash
git add src/companion_bot_core/i18n.py
git commit -m "Make continuity hint tone-aware with warm return examples

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Add post-onboarding hint for first messages

**Files:**
- Modify: `src/companion_bot_core/orchestrator/context_loader.py:249-356`
- Test: `tests/unit/orchestrator/test_context_loader.py`

**Step 1: Write a failing test for post-onboarding hint**

Add a test that verifies when conversation history has 0-2 messages (new user),
the system prompt includes a `[FirstContact]` section with an onboarding hint.

```python
async def test_post_onboarding_hint_injected_for_new_user(
    db_session, snapshot_store, fake_redis,
):
    """First 2 messages after onboarding should get a first-contact hint."""
    user_id = uuid4()
    # No messages in history = brand new user after onboarding
    ctx = await load_user_context(
        db_session, snapshot_store, user_id,
        redis=fake_redis,
    )
    assert "[FirstContact]" in ctx.system_prompt


async def test_no_post_onboarding_hint_for_established_user(
    db_session, snapshot_store, fake_redis,
):
    """Users with 3+ messages should NOT get the first-contact hint."""
    user_id = uuid4()
    # Pre-populate 5 messages
    for i in range(5):
        db_session.add(ConversationMessage(
            user_id=user_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"message {i}",
            model="test",
        ))
    await db_session.flush()

    ctx = await load_user_context(
        db_session, snapshot_store, user_id,
        redis=fake_redis,
    )
    assert "[FirstContact]" not in ctx.system_prompt
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/orchestrator/test_context_loader.py -x -q -k "post_onboarding"`
Expected: FAIL (no `[FirstContact]` section exists yet)

**Step 3: Implement the hint injection in load_user_context**

In `context_loader.py`, after loading `history` (line 327) and before the continuity hint block (line 331), add:

```python
# Inject first-contact hint for new users (0-2 messages = just finished onboarding)
_FIRST_CONTACT_THRESHOLD = 3
if len(history) < _FIRST_CONTACT_THRESHOLD:
    # Extract interest from long-term profile if available
    ltp = extract_section(system_prompt, "Long-term Profile")
    interest = _extract_interest_from_profile(ltp)
    resolved_locale = normalize_locale(locale)
    first_contact = tr(
        "prompt.first_contact_hint", resolved_locale,
        interest=interest,
    )
    system_prompt = f"{system_prompt}\n\n[FirstContact]\n{first_contact}"
```

Add the helper function:

```python
def _extract_interest_from_profile(profile_text: str) -> str:
    """Extract the user's interest from their long-term profile text."""
    for line in profile_text.splitlines():
        stripped = line.strip().lower()
        if "интересуется:" in stripped or "interested in:" in stripped:
            return line.strip().split(":", 1)[-1].strip()
    return ""
```

Add translation key to `i18n.py`:

```python
"prompt.first_contact_hint": {
    "ru": (
        "Это начало общения с новым пользователем. "
        "Покажи свою пользу через конкретный пример, "
        "привязанный к интересу пользователя ({interest}). "
        "НЕ спрашивай «чем помочь?» — предложи сам. "
        "Если интерес не указан — предложи что-нибудь универсальное."
    ),
    "en": (
        "This is the start of a conversation with a new user. "
        "Show your value through a concrete example "
        "related to the user's interest ({interest}). "
        "Do NOT ask 'how can I help?' — proactively suggest something. "
        "If no interest is specified, suggest something universal."
    ),
},
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/orchestrator/test_context_loader.py -x -q -k "post_onboarding"`
Expected: PASS

**Step 5: Run full suite**

Run: `pytest tests/unit/ -x -q && ruff check . && mypy .`
Expected: PASS

**Step 6: Commit**

```bash
git add src/companion_bot_core/orchestrator/context_loader.py src/companion_bot_core/i18n.py tests/
git commit -m "Add post-onboarding first-contact hint for new users

Inject a [FirstContact] prompt section when conversation history
has fewer than 3 messages, guiding the bot to proactively show
value based on user's interests instead of asking 'how can I help?'

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Update welcome-back message

**Files:**
- Modify: `src/companion_bot_core/i18n.py:437-443`

**Step 1: Update the welcome-back messages**

Replace the generic "Чем могу помочь?" with warmer text:

```python
"start.welcome_back": {
    "ru": "👋 О, {name}! Рада видеть снова.",
    "en": "👋 Hey, {name}! Good to see you again.",
},
"start.welcome_back_no_name": {
    "ru": "👋 С возвращением! Как дела?",
    "en": "👋 Welcome back! How's it going?",
},
```

**Step 2: Run tests**

Run: `pytest tests/unit/ -x -q`

**Step 3: Commit**

```bash
git add src/companion_bot_core/i18n.py
git commit -m "Warm up welcome-back messages, drop generic 'how can I help'

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Full verification and deploy

**Step 1: Run complete test suite**

```bash
pytest tests/unit/ -q
pytest tests/integration/ -q
pytest tests/security/ -q
ruff check .
mypy .
```

Expected: All green.

**Step 2: Push and wait for CI + deploy**

```bash
git push origin main
gh run list --limit 1 --json status,conclusion
# Poll until CI passes
gh run list --workflow deploy --limit 1 --json status,conclusion
# Poll until deploy completes
```

---

### Task 6: Launch test-dev-loop

After deploy completes, launch the test cycle from a fresh context.

**Step 1: Start a new Claude session in the test-companion-bot directory**

The tester agent should use the prompt from `test-dev-loop.md` (Tester Agent Prompt section).

Working directory: `/Users/mingazhev/Repos/SideProjects/test-companion-bot`

The tester should:
1. Test all 8 personas (masha, dmitry, anna, oleg, liza, victor, katya, artem)
2. Run return tests on 2-3 personas
3. Generate per-persona reports + summary to `docs/reports/`
4. Include benchmark scores, NPS, anchors/friction counts

**Step 2: Review results**

Read the summary report and decide if another dev round is needed.
