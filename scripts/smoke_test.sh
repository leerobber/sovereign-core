#!/bin/bash
# Smoke test — verifies the full chain is working
# Run after gateway is up: bash scripts/smoke_test.sh

BASE="${1:-http://localhost:8000}"
CYAN='\033[96m'; GREEN='\033[92m'; RED='\033[91m'; BOLD='\033[1m'; RST='\033[0m'

pass=0; fail=0

check() {
    local name="$1"; shift
    local expected_status="${1}"; shift
    local result
    result=$(curl -s -o /tmp/sc_resp -w "%{http_code}" "$@")
    if [ "$result" = "$expected_status" ]; then
        echo -e "  ${GREEN}✓${RST} $name"
        ((pass++)) || true
    else
        echo -e "  ${RED}✗${RST} $name  [got $result, want $expected_status]"
        cat /tmp/sc_resp 2>/dev/null | head -3
        ((fail++)) || true
    fi
}

echo -e "\n${BOLD}Sovereign Core Smoke Test — $BASE${RST}\n"

echo "── Core ─────────────────────────────────────────"
check "GET /health"            200 "$BASE/health"
check "GET /metrics"           200 "$BASE/metrics"
check "GET /docs"              200 "$BASE/docs"

echo ""
echo "── Status ───────────────────────────────────────"
check "GET /status/"           200 "$BASE/status/"
check "GET /status/backends"   200 "$BASE/status/backends"

echo ""
echo "── OpenAI Compat ────────────────────────────────"
check "GET /v1/models"         200 "$BASE/v1/models"
check "POST /v1/chat/completions" 200 \
    -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}'

echo ""
echo "── KAIROS ───────────────────────────────────────"
check "GET /kairos/leaderboard" 200 "$BASE/kairos/leaderboard"
check "GET /kairos/agents"      200 "$BASE/kairos/agents"
check "POST /kairos/sage"       200 \
    -X POST "$BASE/kairos/sage" \
    -H "Content-Type: application/json" \
    -d '{"task":"optimize inference latency on RTX 5050","max_cycles":1}'

echo ""
echo "── Inference ────────────────────────────────────"
check "POST /inference" 200 \
    -X POST "$BASE/inference" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","prompt":"Say hello","options":{"num_predict":5}}'

echo ""
echo -e "${BOLD}Result: ${GREEN}$pass passed${RST}, ${RED}$fail failed${RST}"
[ "$fail" -eq 0 ] && echo -e "${GREEN}All checks passed — gateway fully operational.${RST}" \
                  || echo -e "${RED}Some checks failed — review logs above.${RST}"
