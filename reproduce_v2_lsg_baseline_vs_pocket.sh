#!/usr/bin/env bash
# One-click: LSG Baseline vs Pocket-SLAM on EuRoC V2_01_easy
# Expected: Baseline 6.64cm / 32.50 PSNR ; Pocket 6.38cm / 32.04 PSNR
set -euo pipefail
cd "$(dirname "$0")"
GPU="${1:-0}"
python -u reproduce_v2_lsg_baseline_vs_pocket.py --gpu "$GPU" "$@"
