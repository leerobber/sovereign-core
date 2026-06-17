#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing Ollama ==="

if command -v ollama &>/dev/null; then
    echo "Ollama already installed: $(ollama --version)"
else
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "Ollama installed successfully"
fi

# Start ollama serve in background if not running
if ! pgrep -x ollama &>/dev/null; then
    echo "Starting Ollama server..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# Verify
echo "Ollama endpoint check:"
curl -s http://localhost:11434/api/tags | python3 -m json.tool || echo "Waiting for Ollama to start..."
sleep 2
curl -s http://localhost:11434/api/tags | python3 -m json.tool

echo ""
echo "=== Pulling Llama 3.2 3B ==="
ollama pull llama3.2:3b

echo ""
echo "=== Verification ==="
ollama list
echo ""
echo "Testing inference..."
curl -s http://localhost:11434/api/generate -d '{"model":"llama3.2:3b","prompt":"Say hello in one sentence.","stream":false}' | python3 -m json.tool

echo ""
echo "=== KAN-16 COMPLETE ==="
