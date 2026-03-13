#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Twin] Starting..."
exec python3 "$SCRIPT_DIR/overlay.py"
