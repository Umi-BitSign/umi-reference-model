#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .runtime/bin/python ]]; then
  python3.10 -m venv .runtime
  .runtime/bin/python -m pip install pip==24.0
  .runtime/bin/python -m pip install -r requirements.txt
fi
exec .runtime/bin/python -B run.py "$@"
