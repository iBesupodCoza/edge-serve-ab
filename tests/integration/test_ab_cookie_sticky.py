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


def test_header_override_B_sets_model():
    with TestClient(create_app()) as client:
        b64 = _img_b64()
        r = client.post(
            "/v1/infer", headers={"X-Model-Override": "B"}, json={"image_b64": b64, "img_size": 224}
        )
        assert r.status_code == 200
        assert r.json()["model_used"] == "B"


def test_sticky_cookie_persists_group():
    with TestClient(create_app()) as client:
        b64 = _img_b64()
        r1 = client.post("/v1/infer", json={"image_b64": b64, "img_size": 224})
        assert r1.status_code == 200
        g = r1.json()["model_used"]
        cookie = r1.cookies.get("ab_group")
        assert cookie == g

        # Second call with cookie should stick
        r2 = client.post(
            "/v1/infer", cookies={"ab_group": cookie}, json={"image_b64": b64, "img_size": 224}
        )
        assert r2.status_code == 200
        assert r2.json()["model_used"] == g


def test_admin_config_update_and_warmup():
    with TestClient(create_app()) as client:
        auth = {"Authorization": "Bearer admin"}
        # get default
        g = client.get("/admin/config", headers=auth)
        assert g.status_code == 200
        # set weights (90/10 -> 80/20)
        s = client.post(
            "/admin/config", headers=auth, json={"ab_weight_a": 0.8, "ab_weight_b": 0.2}
        )
        assert s.status_code == 200
        # warmup both models
        w = client.post("/admin/warmup", headers=auth, params={"runs": 1})
        assert w.status_code == 200
