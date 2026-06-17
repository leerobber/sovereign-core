#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source sovereign-env/bin/activate

echo "=== Sovereign Core ==="
echo "Starting API Gateway on http://0.0.0.0:8000"
echo ""

exec uvicorn src.api.gateway:app --host 0.0.0.0 --port 8000 --reload --log-level info
