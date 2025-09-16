from __future__ import annotations

import base64
import io

from PIL import Image
from starlette.testclient import TestClient

from app.server import create_app


def _img_b64() -> str:
    img = Image.new("RGB", (256, 256), (200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_trace_id_generated_and_returned():
    with TestClient(create_app()) as client:
        b64 = _img_b64()
        r = client.post("/v1/infer", json={"image_b64": b64, "img_size": 224})
        assert r.status_code == 200
        # Should always be present
        tid = r.headers.get("Trace-Id")
        assert tid is not None and len(tid) > 0
        # X-Request-ID mirror is also set
        assert r.headers.get("X-Request-ID") == tid


def test_incoming_x_request_id_is_respected_on_simple_route():
    with TestClient(create_app()) as client:
        incoming = "test-fixed-id-123"
        r = client.get("/healthz", headers={"X-Request-ID": incoming})
        assert r.status_code == 200
        assert r.headers.get("Trace-Id") == incoming
        assert r.headers.get("X-Request-ID") == incoming
