#!/usr/bin/env bash
# Sovereign mesh health — WSL-native GH05T3 + sovereign-core gateway.
# Production gateway uses :8000; GH05T3 agent plane uses :8001/:8002.
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

RUNTIME="${GH05T3_RUNTIME:-wsl}"
WIN_HOST="${GH05T3_WINDOWS_HOST:-$(ip route 2>/dev/null | awk '/default/{print $3; exit}')}"
WIN_HOST="${WIN_HOST:-127.0.0.1}"
SOVEREIGN_URL="${SOVEREIGN_GATEWAY_URL:-http://localhost:8000}"

if [[ "$RUNTIME" == "windows" ]]; then
  GH05T3_HOST="$WIN_HOST"
else
  GH05T3_HOST="localhost"
fi

check() {
  local name="$1" url="$2"
  if code=$(curl -s -o /tmp/mesh_body.txt -w "%{http_code}" --max-time 5 "$url" 2>/dev/null) && [ "$code" = "200" ]; then
    echo -e "${GREEN}✓${NC} $name  $url  HTTP $code"
    head -c 120 /tmp/mesh_body.txt; echo
  else
    echo -e "${RED}✗${NC} $name  $url  HTTP ${code:-000}"
  fi
}

echo "=== Sovereign Mesh Health (runtime=$RUNTIME) ==="
check "Ollama"           "http://localhost:11434/api/version"
check "sovereign-core"   "${SOVEREIGN_URL}/health"
check "GH05T3 backend"   "http://${GH05T3_HOST}:8001/api/health"
check "GH05T3 gateway"   "http://${GH05T3_HOST}:8002/health"
echo "---"
if curl -sf --max-time 3 "${SOVEREIGN_URL}/v1/repos" >/tmp/mesh_repos.json 2>/dev/null; then
  echo "Repo registry:"
  python3 -m json.tool /tmp/mesh_repos.json 2>/dev/null | head -40 || cat /tmp/mesh_repos.json
fi