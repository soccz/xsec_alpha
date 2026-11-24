#!/usr/bin/env bash
# Link only the independent user services; do not restart the signal pipeline.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
systemctl --user link "$here/xsec-basket-coverage.service" "$here/xsec-basket-coverage.timer" \
    "$here/xsec-dashboard-delivery.service" "$here/xsec-dashboard-delivery.timer"
systemctl --user daemon-reload
systemctl --user enable --now xsec-basket-coverage.timer xsec-dashboard-delivery.timer
systemctl --user list-timers xsec-basket-coverage.timer xsec-dashboard-delivery.timer --no-pager
