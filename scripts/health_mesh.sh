#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# WSL-native mesh: GH05T3 on localhost. Windows-native: set GH05T3_RUNTIME=windows.
RUNTIME="${GH05T3_RUNTIME:-wsl}"
WIN_HOST="${GH05T3_WINDOWS_HOST:-$(ip route 2>/dev/null | awk '/default/{print $3; exit}')}"
WIN_HOST="${WIN_HOST:-127.0.0.1}"

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
check "sovereign-core"   "http://localhost:8000/health"
check "GH05T3 backend"   "http://${GH05T3_HOST}:8001/api/health"
check "GH05T3 gateway"   "http://${GH05T3_HOST}:8002/health"
echo "---"
echo "Repo registry:"
curl -s --max-time 5 "http://localhost:8000/v1/repos" 2>/dev/null | python3 -m json.tool 2>/dev/null | head -40 || echo "(sovereign-core not running)"