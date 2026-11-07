#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${STRICT_PYTHON:-python}" "$SCRIPT_DIR/runner.py" dispatch "$@"
