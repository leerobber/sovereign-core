"""Pytest configuration for sovereign-core test suite."""
import asyncio

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
