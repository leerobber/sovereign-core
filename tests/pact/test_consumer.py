"""
Consumer pact tests — sovereign-client → sovereign-core-gateway.

Generates pact files in ./pacts/ that define the API contract.
Run:  bash scripts/pact/run.sh consumer
"""
from __future__ import annotations

import os

import pytest
import requests

try:
    from pact import Consumer, Provider, Like, Term
    _PACT_AVAILABLE = True
except ImportError:
    _PACT_AVAILABLE = False

PACT_DIR = os.environ.get("PACT_DIR", "pacts")
MOCK_PORT = 1234

pytestmark = pytest.mark.skipif(not _PACT_AVAILABLE, reason="pact-python not installed")


@pytest.fixture(scope="module")
def pact():
    p = Consumer("sovereign-client").has_pact_with(
        Provider("sovereign-core-gateway"),
        host_name="localhost",
        port=MOCK_PORT,
        pact_dir=PACT_DIR,
        log_dir=os.path.join(PACT_DIR, "logs"),
        log_level="WARNING",
    )
    yield p
    p.stop_service()


def test_health_returns_status(pact):
    expected = {
        "status": Term(r"ok|degraded|unhealthy", "ok"),
    }
    (
        pact.given("the gateway is running")
        .upon_receiving("a GET /health request")
        .with_request("GET", "/health")
        .will_respond_with(200, body=Like(expected))
    )
    with pact:
        resp = requests.get(f"http://localhost:{MOCK_PORT}/health")
    assert resp.status_code == 200
    assert resp.json().get("status") in ("ok", "degraded", "unhealthy")


def test_chat_completions_returns_openai_shape(pact):
    req_body = {
        "model": "sovereign-core",
        "messages": [{"role": "user", "content": "ping"}],
    }
    expected = {
        "id": Like("chatcmpl-abc123"),
        "object": "chat.completion",
        "choices": [
            {
                "index": Like(0),
                "message": {
                    "role": "assistant",
                    "content": Like("pong"),
                },
                "finish_reason": Term(r"stop|length", "stop"),
            }
        ],
        "model": Like("sovereign-core"),
        "usage": {
            "prompt_tokens": Like(3),
            "completion_tokens": Like(1),
            "total_tokens": Like(4),
        },
    }
    (
        pact.given("at least one backend is healthy")
        .upon_receiving("a POST /v1/chat/completions request")
        .with_request(
            "POST",
            "/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body=req_body,
        )
        .will_respond_with(200, body=Like(expected))
    )
    with pact:
        resp = requests.post(
            f"http://localhost:{MOCK_PORT}/v1/chat/completions",
            json=req_body,
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) >= 1


def test_silos_returns_list(pact):
    (
        pact.given("the gateway is configured")
        .upon_receiving("a GET /silos request")
        .with_request("GET", "/silos")
        .will_respond_with(
            200,
            body=Like([{"id": Like("rtx5050"), "device": Like("gpu")}]),
        )
    )
    with pact:
        resp = requests.get(f"http://localhost:{MOCK_PORT}/silos")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
