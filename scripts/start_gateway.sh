#!/bin/bash
# Sovereign Core Gateway — quick start script
# Run from the sovereign-core/ repo root

set -euo pipefail

CYAN='\033[96m'; BOLD='\033[1m'; GREEN='\033[92m'; YELLOW='\033[93m'; RST='\033[0m'

echo -e "${BOLD}Sovereign Core Gateway — Starting${RST}"
echo ""

# 1. Check Python
python3 --version >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }

# 2. Install deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${CYAN}Installing dependencies...${RST}"
    pip install -r requirements.txt --quiet
fi

# 3. Create data dirs
mkdir -p data/kairos/agents data/ledger data/pattern_memory

# 4. Check backends
echo -e "${CYAN}Checking Ollama backends...${RST}"
for port in 8001 8002 8003; do
    if curl -sf "http://localhost:$port/api/tags" >/dev/null 2>&1; then
        echo -e "  ${GREEN}● :$port online${RST}"
    else
        echo -e "  ${YELLOW}○ :$port offline${RST}"
    fi
done
echo ""

# 5. Start gateway
echo -e "${CYAN}Starting gateway on :8000...${RST}"
exec python3 -m uvicorn gateway.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --access-log
