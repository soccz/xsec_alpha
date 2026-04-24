#!/usr/bin/env bash
# F1 systemd installation (2026-04-25).
# Run with: sudo bash deploy/install_f1.sh
# Updates:
#   - xsec-retrain.service  : point to new F1 retrain_pipeline.py (with promotion gate)
#   - xsec-measure.service  : new — daily IC measure + drift check
#   - xsec-measure.timer    : new — 00:00 UTC daily
set -euo pipefail

REPO="/mnt/20t/main/gan_t/xsec_alpha"
UNIT_DIR="/etc/systemd/system"

echo "Installing F1 systemd units…"

cp "$REPO/deploy/xsec-retrain.service" "$UNIT_DIR/xsec-retrain.service"
cp "$REPO/deploy/xsec-measure.service" "$UNIT_DIR/xsec-measure.service"
cp "$REPO/deploy/xsec-measure.timer"   "$UNIT_DIR/xsec-measure.timer"

systemctl daemon-reload
systemctl enable --now xsec-measure.timer

echo ""
echo "Current state:"
systemctl list-timers | grep -i xsec || true
echo ""
echo "Next retrain:"
systemctl list-timers xsec-retrain.timer | head
echo ""
echo "Test the new retrain pipeline (dry-run, no file changes):"
echo "  sudo systemctl start xsec-retrain.service  # uses new pipeline"
echo "  or manually: python scripts/retrain_pipeline.py --dry-run"
echo ""
echo "✅ F1 install complete"
