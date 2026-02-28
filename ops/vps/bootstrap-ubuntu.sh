#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/companion-bot-core}"
DEPLOY_USER="${DEPLOY_USER:-${SUDO_USER:-$USER}}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo bash ops/vps/bootstrap-ubuntu.sh"
  exit 1
fi

apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release ufw fail2ban

install -m 0755 -d /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/docker.asc ]]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
fi

if [[ ! -f /etc/apt/sources.list.d/docker.list ]]; then
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null
fi

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker
systemctl enable --now fail2ban

if id -u "${DEPLOY_USER}" > /dev/null 2>&1; then
  usermod -aG docker "${DEPLOY_USER}"
fi

install -d -m 0755 "${APP_DIR}"
chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "${APP_DIR}"

ufw allow OpenSSH
ufw --force enable

echo "Bootstrap completed."
echo "1) Re-login as ${DEPLOY_USER} (to refresh docker group)."
echo "2) Copy .env.vps.example to ${APP_DIR}/.env and fill secrets."
echo "3) Run one local bootstrap deploy from your machine or trigger GitHub deploy workflow."
