# Companion Bot Core — Multi-User Telegram Companion Bot

A production-grade Telegram bot that gives each user a personalized, evolvable AI persona. Each user's persona is shaped by in-chat preferences and refined asynchronously in the background.

## Architecture overview

- aiogram 3.x handles Telegram updates (polling mode)
- Behavior detector classifies intent on every message (tone change, persona change, safety override)
- Emotion detector classifies user message mood (venting, validation, task, farewell, neutral) and injects mode-specific instructions
- Orchestrator assembles per-user prompt context and calls the model; includes topic tracking, repetition guard, bookmark detection, habit tracking, session tracking, mood journal, and feedback collection
- Refinement worker, TTL sweeper, and proactive check-in scheduler run as asyncio background tasks within the same process as the bot (not separate processes). Stopping the bot stops all of them.
- Refinement worker dequeues jobs, calls a refinement model, and evolves the user's prompt snapshot
- TTL sweeper runs once at startup (to clear rows that expired while the bot was offline), then every hour
- Internal aiohttp service exposes `/internal/refine/{user_id}` and `/internal/detect-change` for operator use
- PostgreSQL stores user data with cascading deletes; Redis handles rate limiting, idempotency, queues, and ephemeral state

## Prerequisites

- Python 3.11+
- PostgreSQL (any recent version)
- Redis 6+
- OpenAI API key (or compatible proxy)

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` and fill in the required values:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather |
| `DATABASE_URL` | PostgreSQL async DSN, e.g. `postgresql+asyncpg://user:pass@localhost/companion_bot_core` |
| `REDIS_URL` | Redis URL, e.g. `redis://localhost:6379/0` |
| `OPENAI_API_KEY` | OpenAI API key (or proxy key) |

Notable optional variables (see `.env.example` for full list):

| Variable | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `gpt-5-mini` | Model for chat responses |
| `REFINEMENT_MODEL` | `gpt-5.2` | Model for background prompt refinement |
| `FIELD_ENCRYPTION_KEY` | (required if `ENCRYPT_SENSITIVE_FIELDS=true`) | Fernet key for field encryption |
| `USE_FAKE_ADAPTERS` | `false` | Enable local dev mode (no real API calls) |
| `INTERNAL_SERVER_HOST` | `127.0.0.1` | Internal service bind address (never expose externally) |
| `INTERNAL_SERVER_PORT` | `8080` | Internal service port |
| `CONVERSATION_TTL_SECONDS` | `604800` | Message retention window (7 days) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Override to route requests to a compatible API proxy |
| `RATE_LIMIT_MESSAGES_PER_MINUTE` | `20` | Per-user message rate limit (sliding window) |
| `RATE_LIMIT_GLOBAL_RPS` | `100` | Global requests-per-second cap across all users |
| `REFINEMENT_CADENCE_SECONDS` | `3600` | Minimum seconds between cadence-triggered refinements per user |
| `REFINEMENT_ACTIVITY_THRESHOLD` | `5` | Messages before an activity-triggered refinement job is enqueued |
| `LOG_LEVEL` | `INFO` | Root log level |
| `LOG_FORMAT` | `json` | Log renderer: `json` for production, `console` for development |
| `DATABASE_POOL_MIN` | `2` | Minimum connections in the async database pool |
| `DATABASE_POOL_MAX` | `10` | Maximum connections in the async database pool |
| `FEEDBACK_SESSION_INTERVAL` | `10` | Ask for feedback every N-th session at farewell |
| `FEEDBACK_COOLDOWN_DAYS` | `7` | Minimum days between feedback requests per user |

To generate a `FIELD_ENCRYPTION_KEY`:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Running

```bash
# Apply database migrations
alembic upgrade head

# Start the bot (polling mode)
companion-bot-core
# Or run directly:
# python -m companion_bot_core.main
```

Webhook mode (`TELEGRAM_WEBHOOK_HOST != ''`) is not yet implemented and will raise `NotImplementedError` at startup. Use polling mode for all current deployments.

## Local development with fake adapters

```bash
USE_FAKE_ADAPTERS=true companion-bot-core
```

Replaces the real OpenAI client with a deterministic fake. The full pipeline runs without valid API credentials. Responses are prefixed with `[Dev mode]`. Never enable in production.

Seed personas available: `friendly` (default), `professional`, `concise`, and skill packs `coding_help`, `study_buddy`.

## Bot commands

| Command | Description |
|---|---|
| `/start` | Welcome message and command list |
| `/profile` | View current settings |
| `/set_language <ru|en>` | Set chat language (`ru` default) |
| `/set_tone <tone>` | Set response tone: `friendly`, `professional`, `playful`, `neutral`, `casual` |
| `/set_persona <name>` | Set persona name (max 64 chars) |
| `/memory_compact_now` | Request immediate prompt refinement |
| `/reset_persona` | Reset persona and tone to defaults |
| `/rollback` | Revert active prompt snapshot to the previous version |
| `/privacy` | Data retention summary |
| `/delete_my_data` | Permanently delete all personal data (DB rows + Redis keys including rate limits, pending changes, abuse blocks) |
| `/mood [week\|month]` | View mood journal timeline (last 7 or 30 days) |
| `/bookmarks` | List saved conversation moments |
| `/bookmarks search <query>` | Search bookmarks by text |
| `/habits` | List active habits with streaks |
| `/habits archive <N>` | Archive a habit by number |
| `/checkin on HH:MM` | Enable daily proactive check-in at specified time |
| `/checkin off` | Disable daily check-in |
| `/checkin quiet HH:MM-HH:MM` | Set quiet hours for check-ins |

By default, bot communication is in Russian. Users can switch to English with `/set_language en`.

## Internal HTTP API

The internal service binds to `127.0.0.1:8080` by default. Do not expose externally — there is no authentication.

- `POST /internal/refine/{user_id}` — enqueue a refinement job; optional body `{"trigger": "string"}`, returns 202 (or 409 if a refinement is already in progress)
- `POST /internal/detect-change` — classify intent; required body `{"text": "string"}`, returns DetectionResult JSON
- `GET /internal/analytics/overview?days=7` — aggregate engagement metrics (active users, session stats, return rate)
- `GET /internal/analytics/users/{user_id}?days=30` — per-user engagement profile
- `GET /metrics` — Prometheus metrics

## Testing

```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/security/
pytest tests/data/
pytest tests/load/
python -m tests.persona.runner  # persona scenario tests (requires LLM)
ruff check .
mypy .
```

## Database migrations

```bash
alembic upgrade head     # apply all migrations
alembic history          # view migration history
```

## Auto-deploy to Ubuntu VPS

This repository includes a ready deployment pipeline:

- Docker image build: [`Dockerfile`](/Users/mingazhev/Repos/SideProjects/TdBot/Dockerfile)
- VPS stack: [`docker-compose.vps.yml`](/Users/mingazhev/Repos/SideProjects/TdBot/docker-compose.vps.yml)
- CI workflow: [`.github/workflows/ci.yml`](/Users/mingazhev/Repos/SideProjects/TdBot/.github/workflows/ci.yml)
- Deploy workflow: [`.github/workflows/deploy.yml`](/Users/mingazhev/Repos/SideProjects/TdBot/.github/workflows/deploy.yml)
- Ubuntu bootstrap script: [`ops/vps/bootstrap-ubuntu.sh`](/Users/mingazhev/Repos/SideProjects/TdBot/ops/vps/bootstrap-ubuntu.sh)

Full setup guide:

- [`docs/deploy-vps-ubuntu.md`](/Users/mingazhev/Repos/SideProjects/TdBot/docs/deploy-vps-ubuntu.md)

## Known limitations

- Webhook mode is not implemented (raises `NotImplementedError` on startup)
