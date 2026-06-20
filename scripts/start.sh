#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
VENV="${SOVEREIGN_VENV:-$PWD/.venv}"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
elif [ -f sovereign-env/bin/activate ]; then
  # shellcheck disable=SC1091
  source sovereign-env/bin/activate
else
  echo "No venv found. Create: python3 -m venv .venv && pip install -e ."
  exit 1
fi

export GH05T3_GATEWAY_URL="${GH05T3_GATEWAY_URL:-http://localhost:8002}"
export GH05T3_RUNTIME="${GH05T3_RUNTIME:-wsl}"

echo "=== Sovereign Core ==="
echo "Starting API Gateway on http://0.0.0.0:8000"
echo "GH05T3 gateway:  $GH05T3_GATEWAY_URL"
echo ""

exec uvicorn src.api.gateway:app --host 0.0.0.0 --port 8000 --reload --log-level info
