import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.server import create_app


def _fake_img_b64(size=224):
    # Simple RGB noise image
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    im = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_infer_top5_ok():
    client = TestClient(create_app())
    # trigger startup (lifespan)
    with client:
        b64 = _fake_img_b64(224)
        r = client.post("/v1/infer", json={"image_b64": b64, "img_size": 224})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_used"] == "A"
        assert isinstance(body["top5"], list) and len(body["top5"]) == 5
        assert isinstance(body["shape"], list) and len(body["shape"]) == 1  # [1000]
