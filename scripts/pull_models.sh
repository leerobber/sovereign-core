#!/bin/bash
# pull_models.sh — Pull all models onto the correct Ollama backends
# Run this ONCE after starting the 3 Ollama instances
# Usage: bash scripts/pull_models.sh

set -euo pipefail

BOLD='\033[1m'; CYAN='\033[96m'; GREEN='\033[92m'; YELLOW='\033[93m'
RED='\033[91m'; DIM='\033[2m'; RST='\033[0m'

# Backend definitions: "NAME PORT MODELS..."
declare -A BACKEND_PORTS=(
    [rtx5050]=8001
    [radeon780m]=8002
    [ryzen7cpu]=8003
)

# Models per backend
declare -A BACKEND_MODELS=(
    [rtx5050]="qwen2.5:14b nemotron-mini"
    [radeon780m]="deepseek-coder:6.7b"
    [ryzen7cpu]="llama3.2:3b"
)

# Labels for display
declare -A BACKEND_LABELS=(
    [rtx5050]="RTX 5050 (8GB VRAM)"
    [radeon780m]="Radeon 780M (4GB VRAM)"
    [ryzen7cpu]="Ryzen 7 (CPU fallback)"
)

pull_model() {
    local port="$1"
    local model="$2"
    local url="http://localhost:$port"

    echo -ne "    Pulling $model ... "
    if curl -sf "$url/api/tags" | python3 -c "
import json,sys
data=json.load(sys.stdin)
models=[m['name'] for m in data.get('models',[])]
print('EXISTS' if any('$model' in m for m in models) else 'MISSING')
" 2>/dev/null | grep -q "EXISTS"; then
        echo -e "${GREEN}already present${RST}"
        return 0
    fi

    # Pull via API
    local result
    result=$(curl -sf -X POST "$url/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"$model\",\"stream\":false}" \
        --max-time 3600 2>&1 || echo "FAILED")

    if echo "$result" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); exit(0 if d.get('status')=='success' else 1)" 2>/dev/null; then
        echo -e "${GREEN}done${RST}"
    else
        echo -e "${YELLOW}may have pulled (check manually)${RST}"
    fi
}

echo -e "\n${BOLD}Sovereign Core — Model Pull Script${RST}"
echo -e "${DIM}Pulls models onto the correct GPU backend Ollama instances${RST}\n"

total_ok=0
total_fail=0

for backend in rtx5050 radeon780m ryzen7cpu; do
    port="${BACKEND_PORTS[$backend]}"
    label="${BACKEND_LABELS[$backend]}"
    models="${BACKEND_MODELS[$backend]}"

    echo -e "${CYAN}── $label (port $port) ──${RST}"

    # Check backend is alive
    if ! curl -sf "http://localhost:$port/api/tags" >/dev/null 2>&1; then
        echo -e "  ${RED}✗ Backend offline — skipping${RST}"
        echo -e "  ${DIM}Start with: OLLAMA_HOST=0.0.0.0:$port ollama serve${RST}"
        echo ""
        ((total_fail++)) || true
        continue
    fi
    echo -e "  ${GREEN}● Backend online${RST}"

    for model in $models; do
        pull_model "$port" "$model"
        ((total_ok++)) || true
    done
    echo ""
done

# Verify via gateway
echo -e "${CYAN}── Verifying via gateway ──${RST}"
if curl -sf "http://localhost:8000/v1/models" >/dev/null 2>&1; then
    echo -e "  ${GREEN}Gateway: online${RST}"
    curl -sf "http://localhost:8000/v1/models" | python3 -c "
import json,sys
data=json.load(sys.stdin)
models=data.get('data',[])
print(f'  Models registered: {len(models)}')
for m in models[:8]:
    print(f'    {m[\"id\"]:40s} [{m[\"backend\"]}]')
" 2>/dev/null || true
else
    echo -e "  ${YELLOW}Gateway not started yet — run: bash scripts/start_gateway.sh${RST}"
fi

echo ""
echo -e "${BOLD}Done. Run the smoke test to verify everything:${RST}"
echo -e "  ${CYAN}bash scripts/smoke_test.sh${RST}"
echo ""
