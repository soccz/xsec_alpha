#!/usr/bin/env bash
# Install the xsec backup as a user-level systemd timer.

set -euo pipefail

REPO="/mnt/20t/main/gan_t/xsec_alpha"
UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"

install -d -m 0755 "${UNIT_DIR}"
install -m 0644 "${REPO}/deploy/xsec-backup.service" \
  "${UNIT_DIR}/xsec-backup.service"
install -m 0644 "${REPO}/deploy/xsec-backup.timer" \
  "${UNIT_DIR}/xsec-backup.timer"

systemctl --user daemon-reload
systemctl --user enable --now xsec-backup.timer

echo "[install_backup] installed and enabled xsec-backup.timer"
systemctl --user --no-pager list-timers xsec-backup.timer
