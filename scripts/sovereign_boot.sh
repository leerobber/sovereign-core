#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║         SOVEREIGN CORE — MASTER BOOT SCRIPT                 ║
# ║         One command. Everything up. Ready to evolve.         ║
# ╚══════════════════════════════════════════════════════════════╝
# Usage: bash scripts/sovereign_boot.sh [start|stop|status|restart]

set -euo pipefail

BOLD='\033[1m'; CYAN='\033[96m'; GREEN='\033[92m'
YELLOW='\033[93m'; RED='\033[91m'; DIM='\033[2m'; RST='\033[0m'
GHOST='\033[95m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

banner() {
    echo -e ""
    echo -e "${GHOST}${BOLD}  ██████╗ ██╗  ██╗ ██████╗ ███████╗████████╗${RST}"
    echo -e "${GHOST}${BOLD}  ██╔════╝██║  ██║██╔═══██╗██╔════╝╚══██╔══╝${RST}"
    echo -e "${GHOST}${BOLD}  ██║  ███╗███████║██║   ██║███████╗   ██║   ${RST}"
    echo -e "${GHOST}${BOLD}  ██║   ██║██╔══██║██║   ██║╚════██║   ██║   ${RST}"
    echo -e "${GHOST}${BOLD}  ╚██████╔╝██║  ██║╚██████╔╝███████║   ██║   ${RST}"
    echo -e "${GHOST}${BOLD}   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝  ${RST}"
    echo -e "${DIM}  Sovereign Core — Heterogeneous Compute Gateway${RST}"
    echo -e "${DIM}  Built by Robert 'Terry' Lee Jr. | SovereignNation LLC${RST}"
    echo -e ""
}

# ── STEP 1: Check Ollama is installed ─────────────────────────────────────────
check_ollama() {
    if ! command -v ollama &>/dev/null; then
        echo -e "${RED}✗ Ollama not found. Install from https://ollama.ai${RST}"
        exit 1
    fi
    echo -e "${GREEN}✅ Ollama found: $(ollama --version 2>/dev/null | head -1)${RST}"
}

# ── STEP 2: Check if single Ollama is running (user said ollama serve is up) ──
check_existing_ollama() {
    echo -e "\n${CYAN}── Checking Ollama instances ──${RST}"

    # If user already has ollama serve running on default :11434
    if curl -sf "http://localhost:11434/api/tags" >/dev/null 2>&1; then
        echo -e "${GREEN}  ✅ Ollama running on :11434 (default)${RST}"
        OLLAMA_DEFAULT_RUNNING=1
    else
        OLLAMA_DEFAULT_RUNNING=0
    fi

    # Check dedicated backend ports
    local any_backend=0
    for port in 8001 8002 8003; do
        if curl -sf "http://localhost:$port/api/tags" >/dev/null 2>&1; then
            echo -e "${GREEN}  ✅ Backend on :$port already running${RST}"
            any_backend=1
        fi
    done

    # If backends not on dedicated ports — start them
    if [ "$any_backend" -eq 0 ]; then
        echo -e "${YELLOW}  ⚡ No dedicated backends found — starting now...${RST}"
        start_backends
    else
        echo -e "${GREEN}  ✅ Backends already up — skipping start${RST}"
    fi
}

# ── STEP 3: Start 3 Ollama backend instances ──────────────────────────────────
start_backends() {
    echo -e "\n${CYAN}── Starting Ollama Backends ──${RST}"

    start_one_backend() {
        local name="$1" port="$2" gpu_env="$3" label="$4"
        echo -ne "  ${CYAN}$label (:$port)...${RST} "
        if curl -sf "http://localhost:$port/api/tags" >/dev/null 2>&1; then
            echo -e "${GREEN}already up${RST}"; return 0
        fi
        eval "env OLLAMA_HOST=0.0.0.0:$port $gpu_env ollama serve \
            > $LOG_DIR/ollama_${name}.log 2>&1 &"
        echo $! > "$LOG_DIR/ollama_${name}.pid"
        # Wait up to 20s
        for i in $(seq 1 20); do
            sleep 1
            if curl -sf "http://localhost:$port/api/tags" >/dev/null 2>&1; then
                echo -e "${GREEN}✅ started (pid $(cat $LOG_DIR/ollama_${name}.pid))${RST}"
                return 0
            fi
        done
        echo -e "${YELLOW}⚠️  timeout — check $LOG_DIR/ollama_${name}.log${RST}"
    }

    # RTX 5050 = NVIDIA CUDA
    start_one_backend "rtx5050" 8001 \
        "CUDA_VISIBLE_DEVICES=0" \
        "RTX 5050   (NVIDIA 8GB)"

    # Radeon 780M = AMD ROCm
    start_one_backend "radeon780m" 8002 \
        "HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0" \
        "Radeon 780M (AMD 4GB) "

    # Ryzen 7 = CPU fallback
    start_one_backend "ryzen7cpu" 8003 \
        "OLLAMA_NO_GPU=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1" \
        "Ryzen 7    (CPU fallback)"
}

# ── STEP 4: Ensure models are loaded ──────────────────────────────────────────
check_models() {
    echo -e "\n${CYAN}── Model Check ──${RST}"

    check_one() {
        local port="$1" model="$2" label="$3"
        if ! curl -sf "http://localhost:$port/api/tags" >/dev/null 2>&1; then
            echo -e "  ${RED}✗ :$port offline — cannot check $model${RST}"; return
        fi
        local present
        present=$(curl -sf "http://localhost:$port/api/tags" | \
            python3 -c "
import json,sys
d=json.load(sys.stdin)
names=[m['name'] for m in d.get('models',[])]
print('YES' if any('${model%%:*}' in n for n in names) else 'NO')
" 2>/dev/null || echo "NO")
        if [ "$present" = "YES" ]; then
            echo -e "  ${GREEN}✅ $label — $model${RST}"
        else
            echo -e "  ${YELLOW}⚡ $label — pulling $model...${RST}"
            curl -sf -X POST "http://localhost:$port/api/pull" \
                -H "Content-Type: application/json" \
                -d "{\"name\":\"$model\",\"stream\":false}" \
                --max-time 3600 > /dev/null 2>&1 && \
                echo -e "  ${GREEN}  ✅ pulled${RST}" || \
                echo -e "  ${YELLOW}  ⚠️  pull may have failed — run: bash scripts/pull_models.sh${RST}"
        fi
    }

    check_one 8001 "qwen2.5:14b"      "RTX 5050   "
    check_one 8002 "deepseek-coder:6.7b" "Radeon 780M"
    check_one 8003 "llama3.2:3b"      "Ryzen 7 CPU"
}

# ── STEP 5: Start the FastAPI gateway ─────────────────────────────────────────
start_gateway() {
    echo -e "\n${CYAN}── Starting Sovereign Gateway ──${RST}"

    # Check if already running
    if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
        echo -e "${GREEN}  ✅ Gateway already running on :8000${RST}"
        return 0
    fi

    cd "$REPO_ROOT"

    # Install deps if needed
    if ! python3 -c "import fastapi,uvicorn" 2>/dev/null; then
        echo -e "  ${CYAN}Installing Python deps...${RST}"
        pip install -r requirements.txt -q
    fi

    # Create data dirs
    mkdir -p data/kairos/agents data/ledger data/pattern_memory

    # Start gateway in background
    echo -e "  ${CYAN}Launching gateway on :8000...${RST}"
    nohup python3 -m uvicorn gateway.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info \
        > "$LOG_DIR/gateway.log" 2>&1 &
    echo $! > "$LOG_DIR/gateway.pid"

    # Wait up to 15s
    for i in $(seq 1 15); do
        sleep 1
        if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
            echo -e "  ${GREEN}✅ Gateway online :8000 (pid $(cat $LOG_DIR/gateway.pid))${RST}"
            return 0
        fi
    done
    echo -e "  ${RED}✗ Gateway failed to start — check $LOG_DIR/gateway.log${RST}"
    echo -e "  ${DIM}Try manually: python3 -m uvicorn gateway.main:app --host 0.0.0.0 --port 8000${RST}"
}

# ── STEP 6: Final status report ───────────────────────────────────────────────
status_report() {
    echo -e "\n${BOLD}╔══════════════════════════════════════════════╗"
    echo -e "║         SOVEREIGN CORE — STATUS              ║"
    echo -e "╚══════════════════════════════════════════════╝${RST}"

    declare -A LABELS=(
        [8001]="RTX 5050   (Qwen2.5:14b)     "
        [8002]="Radeon 780M (DeepSeek-Coder)  "
        [8003]="Ryzen 7    (Llama3.2:3b)      "
        [8000]="Sovereign Gateway             "
    )

    for port in 8001 8002 8003 8000; do
        local url="http://localhost:$port"
        local endpoint="/api/tags"
        [ "$port" = "8000" ] && endpoint="/health"
        if curl -sf "${url}${endpoint}" >/dev/null 2>&1; then
            echo -e "  ${GREEN}●${RST} ${LABELS[$port]} :$port  ${GREEN}ONLINE${RST}"
        else
            echo -e "  ${RED}○${RST} ${LABELS[$port]} :$port  ${RED}OFFLINE${RST}"
        fi
    done

    echo ""
    # Show available models via gateway if online
    if curl -sf "http://localhost:8000/v1/models" >/dev/null 2>&1; then
        echo -e "${CYAN}  Registered models:${RST}"
        curl -sf "http://localhost:8000/v1/models" | python3 -c "
import json,sys
data=json.load(sys.stdin)
for m in data.get('data',[]):
    print(f\"    ✅ {m['id']:35s} [{m.get('backend','?')}]\")
" 2>/dev/null || true
    fi

    echo -e "\n${DIM}  Logs: $LOG_DIR/${RST}"
    echo -e "${DIM}  Gateway log: $LOG_DIR/gateway.log${RST}"
    echo -e "\n${GREEN}${BOLD}  Sovereign Core is live. KAIROS ready.${RST}\n"
}

# ── STOP ──────────────────────────────────────────────────────────────────────
stop_all() {
    echo -e "\n${CYAN}Stopping all Sovereign Core processes...${RST}"
    for name in gateway ollama_rtx5050 ollama_radeon780m ollama_ryzen7cpu; do
        pid_file="$LOG_DIR/${name}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            kill "$pid" 2>/dev/null && \
                echo -e "  ${GREEN}✅ Stopped $name (pid $pid)${RST}" || \
                echo -e "  ${DIM}  $name already stopped${RST}"
            rm -f "$pid_file"
        fi
    done
    echo -e "${GREEN}Done.${RST}\n"
}

# ── MAIN ──────────────────────────────────────────────────────────────────────
case "${1:-start}" in
    start)
        banner
        check_ollama
        check_existing_ollama
        check_models
        start_gateway
        status_report
        ;;
    stop)
        stop_all
        ;;
    status)
        banner
        status_report
        ;;
    restart)
        stop_all
        sleep 2
        exec "$0" start
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
