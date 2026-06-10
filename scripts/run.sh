#!/usr/bin/env bash
set -euo pipefail
echo "HLS Alpha Engine — Startup"
uv venv --python 3.13 2>/dev/null || true
source .venv/bin/activate
uv pip install -e ".[dev]" --quiet
hls-alpha hls-run --n-assets 8 --n-periods 2000
