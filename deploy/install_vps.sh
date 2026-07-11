#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/omniquant/app}"
VENV_DIR="${VENV_DIR:-/opt/omniquant/venv}"
SERVICE_USER="${SERVICE_USER:-omniquant}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "请用 root 执行：sudo bash deploy/install_vps.sh"
  exit 1
fi

apt-get update
apt-get install -y python3 python3-venv python3-pip nginx certbot python3-certbot-nginx curl git

if ! swapon --show | grep -q .; then
  fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

id -u "${SERVICE_USER}" >/dev/null 2>&1 || useradd --system --home /opt/omniquant --shell /usr/sbin/nologin "${SERVICE_USER}"

mkdir -p /opt/omniquant /var/cache/omniquant /etc/omniquant
chown -R "${SERVICE_USER}:${SERVICE_USER}" /opt/omniquant /var/cache/omniquant

if [[ ! -f /etc/omniquant/omniquant.env ]]; then
  cp "${APP_DIR}/deploy/omniquant.env.example" /etc/omniquant/omniquant.env
  chmod 640 /etc/omniquant/omniquant.env
  chown root:"${SERVICE_USER}" /etc/omniquant/omniquant.env
fi

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip wheel setuptools
"${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt"

cp "${APP_DIR}/deploy/omniquant-api.service" /etc/systemd/system/omniquant-api.service
systemctl daemon-reload
systemctl enable omniquant-api
systemctl restart omniquant-api

echo "OmniQuant API 已安装并启动。检查状态：systemctl status omniquant-api --no-pager"
echo "下一步：复制 deploy/nginx.omniquant.conf 到 /etc/nginx/sites-available/，替换域名，然后运行 certbot --nginx。"
