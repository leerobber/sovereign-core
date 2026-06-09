#!/bin/bash
# pull_models.sh — Pull all models onto correct Ollama backends
# Models as of 2026-04-17 — already installed on this machine
# Run only if you've wiped Ollama or moved to a new machine

set -euo pipefail
BOLD='\033[1m'; CYAN='\033[96m'; GREEN='\033[92m'; YELLOW='\033[93m'; RED='\033[91m'; RST='\033[0m'

echo -e "\n${BOLD}Sovereign Core — Model Inventory${RST}"
echo -e "${CYAN}Checking what's installed on each backend...${RST}\n"

check_backend() {
    local port="$1" label="$2"
    echo -e "${CYAN}── $label (:$port) ──${RST}"
    if ! curl -sf "http://localhost:$port/api/tags" >/dev/null 2>&1; then
        echo -e "  ${RED}✗ Offline${RST}"; echo ""; return
    fi
    curl -sf "http://localhost:$port/api/tags" | python3 -c "
import json,sys
d=json.load(sys.stdin)
models=d.get('models',[])
if not models:
    print('  (no models loaded yet)')
for m in models:
    size=m.get('size',0)
    gb=size/1e9
    print(f'  ✅ {m[\"name\"]:<35} {gb:.1f}GB')
" 2>/dev/null
    echo ""
}

check_backend 8001 "RTX 5050   (NVIDIA 8GB)  → gemma3:12b primary"
check_backend 8002 "Radeon 780M (AMD 4GB)   → qwen2.5:7b primary"
check_backend 8003 "Ryzen 7    (CPU)         → llama3.2:3b primary"

echo -e "${CYAN}── Recommended model routing ──${RST}"
echo -e "  RTX 5050   :8001 → gemma3:12b, dolphin-llama3:8b, llama3:latest, nomic-embed-text"
echo -e "  Radeon 780M :8002 → qwen2.5:7b, dolphin-phi"
echo -e "  Ryzen 7    :8003 → llama3.2:3b, dolphin-phi"
echo ""
echo -e "${CYAN}To load a specific model onto a backend:${RST}"
echo -e "  curl -X POST http://localhost:8001/api/pull -d '{"name":"gemma3:12b"}'"
echo -e "  curl -X POST http://localhost:8002/api/pull -d '{"name":"qwen2.5:7b"}'"
echo -e "  curl -X POST http://localhost:8003/api/pull -d '{"name":"llama3.2:3b"}'"
echo ""
