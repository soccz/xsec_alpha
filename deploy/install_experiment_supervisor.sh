#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/20t/main/gan_t/xsec_alpha"
UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"

test -f "${REPO}/output/experiment_supervision/review_plan.json"
install -d -m 0755 "${UNIT_DIR}"
install -m 0644 "${REPO}/deploy/xsec-experiment-supervisor.service" "${UNIT_DIR}/xsec-experiment-supervisor.service"
install -m 0644 "${REPO}/deploy/xsec-experiment-supervisor.timer" "${UNIT_DIR}/xsec-experiment-supervisor.timer"
systemctl --user daemon-reload
systemctl --user enable --now xsec-experiment-supervisor.timer
systemctl --user start xsec-experiment-supervisor.service
systemctl --user --no-pager list-timers xsec-experiment-supervisor.timer
