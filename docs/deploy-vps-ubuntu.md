# Auto Deploy To Ubuntu VPS

This project is configured for automatic deployment via GitHub Actions.

## 1. One-time VPS bootstrap

Run on the VPS (Ubuntu 22.04/24.04):

```bash
sudo bash ops/vps/bootstrap-ubuntu.sh
```

What it does:
- installs Docker Engine + Docker Compose plugin
- enables Docker service
- enables `ufw` + `fail2ban`
- creates `/opt/companion-bot-core`

## 2. Prepare runtime env on VPS

On VPS:

```bash
cd /opt/companion-bot-core
cp /path/to/repo/.env.vps.example .env
nano .env
```

Required to set at minimum:
- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `POSTGRES_PASSWORD`
- `DATABASE_URL` (must match postgres credentials)

## 3. Add GitHub repository secrets

In `Settings -> Secrets and variables -> Actions -> Secrets`:

- `VPS_HOST` — VPS public IP/domain
- `VPS_USER` — deploy user (must have docker access)
- `VPS_SSH_KEY` — private SSH key for `VPS_USER`
- `GHCR_READ_TOKEN` — GitHub PAT with `read:packages`

## 4. Deployment flow

- CI (`.github/workflows/ci.yml`) runs on each push/PR.
- Deploy (`.github/workflows/deploy.yml`) runs on:
  - push to `main`
  - manual `workflow_dispatch`

Deploy workflow does:
1. Builds Docker image.
2. Pushes image to GHCR (`ghcr.io/<owner>/companion-bot-core:<sha>` and `:latest`).
3. Uploads `docker-compose.vps.yml` to VPS.
4. On VPS: starts `postgres` + `redis`, runs `alembic upgrade head`, restarts bot.

## 5. First run checklist

After setting secrets and `.env`:

1. Merge to `main` (or run workflow manually).
2. Verify on VPS:

```bash
cd /opt/companion-bot-core
docker compose -f docker-compose.vps.yml ps
docker compose -f docker-compose.vps.yml logs -f bot
```

## Notes

- Webhook mode is not implemented, use polling mode (`TELEGRAM_WEBHOOK_HOST=` empty).
- Internal HTTP service stays inside container (`127.0.0.1:8080`), not exposed publicly.
