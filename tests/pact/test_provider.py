"""
Provider pact verification — verifies sovereign-core-gateway satisfies consumer pacts.

Reads pact files from PACT_DIR and replays them against the running provider.
Run:  PROVIDER_BASE_URL=http://localhost:8000 bash scripts/pact/run.sh provider
"""
from __future__ import annotations

import os

import pytest

try:
    from pact import Verifier
    _PACT_AVAILABLE = True
except ImportError:
    _PACT_AVAILABLE = False

PROVIDER_BASE_URL = os.environ.get("PROVIDER_BASE_URL", "http://localhost:8000")
PACT_DIR = os.environ.get("PACT_DIR", "pacts")

pytestmark = pytest.mark.skipif(not _PACT_AVAILABLE, reason="pact-python not installed")


def test_sovereign_core_gateway_satisfies_consumer_pacts():
    """Replay all consumer pact interactions against the running provider."""
    verifier = Verifier(
        provider="sovereign-core-gateway",
        provider_base_url=PROVIDER_BASE_URL,
    )
    # Provider setup headers — pass API key if configured
    provider_headers = []
    api_key = os.environ.get("GATEWAY_API_KEY")
    if api_key:
        provider_headers.append(f"Authorization: Bearer {api_key}")

    output, _ = verifier.verify_pacts(
        PACT_DIR,
        verbose=False,
        provider_states_setup_url=None,
        headers=provider_headers if provider_headers else None,
    )
    assert output == 0, (
        f"Provider pact verification failed (exit code {output}). "
        f"Provider: {PROVIDER_BASE_URL}, Pact dir: {PACT_DIR}"
    )
