#!/usr/bin/env bash
# One-shot sudo install for the 2026-10-05 monitoring hardening of xsec_alpha.
#   sudo bash deploy/install_monitoring_20261005.sh
# - system units: alpha gets StartLimit (3 failures/h -> failed -> its existing OnFailure finally fires);
#   retrain/measure get OnFailure -> xsec-unit-failure@.service (Telegram notice, 6 h dedupe)
# - inotify watch limit 204800 -> 524288 (exhaustion co-caused the 10-03 supervisor timeout)
# - redact a stale Telegram-token-shaped string from the root-owned logs/systemd.log
# The retrain dry-run drop-in (/etc/systemd/system/xsec-retrain.service.d/hold.conf) is left untouched.
set -euo pipefail
X=/mnt/20t/main/gan_t/xsec_alpha
[ "$(id -u)" = 0 ] || { echo "run with: sudo bash $0"; exit 1; }
if systemctl is-active --quiet xsec-alpha.service; then echo "xsec-alpha is running now; retry in a few minutes"; exit 1; fi

for unit in xsec-alpha.service xsec-retrain.service xsec-measure.service xsec-unit-failure@.service; do
  install -m 0644 "$X/deploy/$unit" "/etc/systemd/system/$unit"
done
printf 'fs.inotify.max_user_watches=524288\n' > /etc/sysctl.d/60-xsec-inotify.conf
sysctl -q -p /etc/sysctl.d/60-xsec-inotify.conf
sed -i -E 's/[0-9]{8,10}:[A-Za-z0-9_-]{35}/<redacted-token>/g' "$X/logs/systemd.log"
systemctl daemon-reload

echo "--- verify ---"
echo "alpha:   $(systemctl show xsec-alpha.service -p StartLimitBurst --value) starts/$(systemctl show xsec-alpha.service -p StartLimitIntervalUSec --value), OnFailure=$(systemctl show xsec-alpha.service -p OnFailure --value)"
echo "retrain: OnFailure=$(systemctl show xsec-retrain.service -p OnFailure --value), dry-run kept: $(systemctl show xsec-retrain.service -p ExecStart | grep -c -- --dry-run)"
echo "measure: OnFailure=$(systemctl show xsec-measure.service -p OnFailure --value)"
echo "inotify: $(cat /proc/sys/fs/inotify/max_user_watches)"
echo "token-shaped strings left in systemd.log: $(grep -cE '[0-9]{8,10}:[A-Za-z0-9_-]{35}' "$X/logs/systemd.log" || true)"
systemctl list-timers --all --no-pager | grep -E 'xsec-(alpha|retrain|measure)' || true
echo "done"
