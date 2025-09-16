import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.server import create_app


def _img_b64(size=224):
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    im = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_payload_limit_413(monkeypatch):
    # Tiny limit (1 KB) to force 413; use 'content=' to set Content-Length
    monkeypatch.setenv("MAX_BODY_BYTES", "1024")
    with TestClient(create_app()) as client:
        big = b"x" * 200_000  # 200 KB
        r = client.post("/v1/infer", content=big, headers={"content-type": "application/json"})
        assert r.status_code == 413


def test_rate_limit_429(monkeypatch):
    # Make the limiter deterministic: no refill, 1-token burst => second call must 429
    monkeypatch.setenv("RATE_LIMIT_RPS", "0")
    monkeypatch.setenv("RATE_LIMIT_BURST", "1")
    with TestClient(create_app()) as client:
        b64 = _img_b64()
        ok1 = client.post("/v1/infer", json={"image_b64": b64, "img_size": 224})
        r2 = client.post("/v1/infer", json={"image_b64": b64, "img_size": 224})
        assert ok1.status_code == 200
        assert r2.status_code == 429
