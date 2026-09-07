#!/bin/bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  printf '%s\n' 'Run with sudo: sudo bash deploy/install_telegram_reporting.sh'
  exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
install -m 644 deploy/xsec-alpha.service deploy/xsec-telegram-retry.service \
  deploy/xsec-telegram-retry.timer deploy/xsec-telegram-failure.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now xsec-telegram-retry.timer
systemctl status xsec-telegram-retry.timer --no-pager
