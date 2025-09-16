# tests/unit/test_health.py
from fastapi.testclient import TestClient

from app.server import create_app


def test_health_ok():
    with TestClient(create_app()) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_ready_live():
    # Lifespan runs only inside context manager → models load & warmup
    with TestClient(create_app()) as client:
        r = client.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["models_loaded"] is True
