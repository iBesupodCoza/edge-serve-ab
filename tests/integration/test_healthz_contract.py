from __future__ import annotations

from starlette.testclient import TestClient

from app.server import create_app


def test_healthz_is_ok():
    with TestClient(create_app()) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"ok": True}


def test_readyz_is_ok():
    # Our readiness check is based on inferencers existing on app.state
    with TestClient(create_app()) as client:
        r = client.get("/readyz")
        assert r.status_code == 200
        assert r.json().get("ready") is True
