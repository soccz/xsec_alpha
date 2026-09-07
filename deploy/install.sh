#!/bin/bash
# Install xsec_alpha systemd timers
set -e

echo "Installing xsec_alpha timers..."

# 6h ranking timer
sudo cp deploy/xsec-alpha.service /etc/systemd/system/
sudo cp deploy/xsec-alpha.timer /etc/systemd/system/
sudo cp deploy/xsec-telegram-retry.service deploy/xsec-telegram-retry.timer deploy/xsec-telegram-failure.service /etc/systemd/system/

# Weekly retrain timer
sudo cp deploy/xsec-retrain.service /etc/systemd/system/
sudo cp deploy/xsec-retrain.timer /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now xsec-alpha.timer
sudo systemctl enable --now xsec-retrain.timer
sudo systemctl enable --now xsec-telegram-retry.timer

echo ""
echo "Installed:"
systemctl list-timers | grep xsec
echo ""
echo "Test run: sudo systemctl start xsec-alpha.service"
