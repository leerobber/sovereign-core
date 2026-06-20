#!/usr/bin/env bash
# Pact contract testing entrypoint for sovereign-core
# Usage: bash scripts/pact/run.sh [consumer|provider|publish|can-i-deploy]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACT_DIR="$REPO_ROOT/pacts"
VENV="$REPO_ROOT/.venv"

BROKER_URL="${PACT_BROKER_URL:-}"
BROKER_TOKEN="${PACT_BROKER_TOKEN:-}"
STAGING_URL="${STAGING_BASE_URL:-http://localhost:8000}"
APP_VERSION="${APP_VERSION:-$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo dev)}"
PACTICIPANT="sovereign-core-gateway"

mkdir -p "$PACT_DIR"

# Activate venv if it exists
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
elif [[ -f "$VENV/Scripts/activate" ]]; then
    source "$VENV/Scripts/activate"
fi

CMD="${1:-help}"

case "$CMD" in

  consumer)
    echo "=== Running consumer pact tests ==="
    PACT_DIR="$PACT_DIR" pytest "$REPO_ROOT/tests/pact/test_consumer.py" \
        -v --tb=short
    echo "Pacts written to: $PACT_DIR"
    ;;

  provider)
    echo "=== Running provider pact verification ==="
    PROVIDER_BASE_URL="${STAGING_URL}" \
    PACT_DIR="$PACT_DIR" \
    pytest "$REPO_ROOT/tests/pact/test_provider.py" \
        -v --tb=short
    ;;

  publish)
    echo "=== Publishing pacts to broker ==="
    if [[ -z "$BROKER_URL" || -z "$BROKER_TOKEN" ]]; then
        echo "ERROR: PACT_BROKER_URL and PACT_BROKER_TOKEN must be set"
        exit 1
    fi
    pact-broker publish "$PACT_DIR" \
        --broker-base-url "$BROKER_URL" \
        --broker-token "$BROKER_TOKEN" \
        --consumer-app-version "$APP_VERSION" \
        --branch "$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo main)" \
        --tag "$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo main)"
    echo "Published version $APP_VERSION"
    ;;

  can-i-deploy)
    echo "=== Can-I-Deploy check ==="
    if [[ -z "$BROKER_URL" || -z "$BROKER_TOKEN" ]]; then
        echo "ERROR: PACT_BROKER_URL and PACT_BROKER_TOKEN must be set"
        exit 1
    fi
    pact-broker can-i-deploy \
        --broker-base-url "$BROKER_URL" \
        --broker-token "$BROKER_TOKEN" \
        --pacticipant "$PACTICIPANT" \
        --version "$APP_VERSION" \
        --to-environment production
    ;;

  help|*)
    echo "Usage: bash scripts/pact/run.sh [consumer|provider|publish|can-i-deploy]"
    echo ""
    echo "  consumer      — generate pact files via consumer tests"
    echo "  provider      — verify pacts against running provider"
    echo "  publish       — push pacts to PactFlow broker"
    echo "  can-i-deploy  — check if safe to deploy to production"
    echo ""
    echo "Required env vars for publish/can-i-deploy:"
    echo "  PACT_BROKER_URL    — PactFlow account URL"
    echo "  PACT_BROKER_TOKEN  — PactFlow read/write token"
    echo "  STAGING_BASE_URL   — provider URL for verification (default: http://localhost:8000)"
    ;;

esac
