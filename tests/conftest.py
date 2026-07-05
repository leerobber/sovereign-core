"""Pytest configuration for sovereign-core test suite."""
import asyncio
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _ensure_event_loop():
    """Ensure there is a current event loop for sync tests that call asyncio.get_event_loop()."""
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield


@pytest.fixture(autouse=True)
def bypass_license_gate():
    """
    Forces Sovereign Core into a deterministic dev-mode for tests by patching
    the LicenseEngine.dev_mode_check method. Avoids import-time side effects
    and external license checks during CI.
    """
    from ghost_protocol.fortress import sovereign_security

    with patch.object(sovereign_security.LicenseEngine, "dev_mode_check", return_value=(True, "DEV_MODE")):
        yield
