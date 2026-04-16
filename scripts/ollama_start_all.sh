#!/bin/bash
# ollama_start_all.sh — Start 3 Ollama instances, one per compute backend
# Run this before start_gateway.sh
# Each instance binds to a specific port.

set -euo pipefail

BOLD='\033[1m'; CYAN='\033[96m'; GREEN='\033[92m'; YELLOW='\033[93m'; DIM='\033[2m'; RST='\033[0m'

OLLAMA_BIN="${OLLAMA_BIN:-ollama}"

# Log dir
mkdir -p logs

start_backend() {
    local name="$1"
    local port="$2"
    local gpu_hint="$3"   # cuda:0, rocm:0, cpu
    local label="$4"

    echo -ne "${CYAN}Starting $label (port $port)...${RST} "

    # Check if already running
    if curl -sf "http://localhost:$port/api/tags" >/dev/null 2>&1; then
        echo -e "${GREEN}already running${RST}"
        return 0
    fi

    # Build env
    local env_vars="OLLAMA_HOST=0.0.0.0:$port"

    # GPU selection hints
    case "$gpu_hint" in
        cuda:0)
            env_vars="$env_vars CUDA_VISIBLE_DEVICES=0"
            ;;
        rocm:0)
            env_vars="$env_vars HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0"
            ;;
        cpu)
            env_vars="$env_vars OLLAMA_NO_GPU=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1"
            ;;
    esac

    # Start in background
    eval "env $env_vars $OLLAMA_BIN serve > logs/ollama_${name}.log 2>&1 &"
    local pid=$!
    echo "$pid" > "logs/ollama_${name}.pid"

    # Wait for it to come up (max 15s)
    local tries=0
    while [ $tries -lt 15 ]; do
        sleep 1
        if curl -sf "http://localhost:$port/api/tags" >/dev/null 2>&1; then
            echo -e "${GREEN}started (pid $pid)${RST}"
            return 0
        fi
        ((tries++)) || true
    done

    echo -e "${YELLOW}timeout — check logs/ollama_${name}.log${RST}"
    return 1
}

stop_all() {
    echo -e "${CYAN}Stopping all Ollama backends...${RST}"
    for name in rtx5050 radeon780m ryzen7cpu; do
        pid_file="logs/ollama_${name}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            kill "$pid" 2>/dev/null && echo -e "  Stopped $name (pid $pid)" || true
            rm -f "$pid_file"
        fi
    done
}

case "${1:-start}" in
    start)
        echo -e "\n${BOLD}Starting Sovereign Core Ollama Backends${RST}\n"

        start_backend "rtx5050"    8001 "cuda:0" "RTX 5050   (NVIDIA, 8GB VRAM)"
        start_backend "radeon780m" 8002 "rocm:0" "Radeon 780M (AMD,    4GB VRAM)"
        start_backend "ryzen7cpu"  8003 "cpu"    "Ryzen 7    (CPU fallback     )"

        echo ""
        echo -e "${GREEN}All backends started.${RST}"
        echo -e "${DIM}Next steps:${RST}"
        echo -e "  ${CYAN}bash scripts/pull_models.sh${RST}     # pull models onto each backend"
        echo -e "  ${CYAN}bash scripts/start_gateway.sh${RST}   # start the gateway"
        echo -e "  ${CYAN}bash scripts/smoke_test.sh${RST}      # verify everything"
        echo ""
        ;;

    stop)
        stop_all
        ;;

    status)
        echo -e "\n${BOLD}Backend Status${RST}\n"
        for backend port label in \
            "rtx5050 8001 RTX 5050" \
            "radeon780m 8002 Radeon 780M" \
            "ryzen7cpu 8003 Ryzen 7 CPU"; do
            read -r name p lbl <<< "$backend $port $label"
            if curl -sf "http://localhost:$p/api/tags" >/dev/null 2>&1; then
                model_count=$(curl -sf "http://localhost:$p/api/tags" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null || echo "?")
                echo -e "  ${GREEN}●${RST} $lbl [:$p] — $model_count model(s)"
            else
                echo -e "  ${YELLOW}○${RST} $lbl [:$p] — offline"
            fi
        done
        echo ""
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
