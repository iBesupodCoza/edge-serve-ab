#!/usr/bin/env python
import base64
import io

import httpx
import numpy as np
from PIL import Image

URL = "http://127.0.0.1:8080/v1/infer"
MET = "http://127.0.0.1:8080/metrics"


def make_img_b64(size=224):
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    im = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def infer_once(override=None, client=None):
    hdr = {}
    if override:
        hdr["X-Model-Override"] = override
    b64 = make_img_b64(224)
    if client is None:
        r = httpx.post(URL, json={"image_b64": b64, "img_size": 224}, headers=hdr, timeout=10.0)
    else:
        r = client.post(URL, json={"image_b64": b64, "img_size": 224}, headers=hdr, timeout=10.0)
    r.raise_for_status()
    return r.json(), r.cookies


def grep_metrics():
    txt = httpx.get(MET, timeout=10.0).text
    ab = "\n".join([ln for ln in txt.splitlines() if ln.startswith("ab_assignments_total")])
    sh = "\n".join([ln for ln in txt.splitlines() if ln.startswith("shadow_requests_total")])
    return ab, sh


if __name__ == "__main__":
    print("=== Forced B x3 (header override) ===")
    for _ in range(3):
        body, _ = infer_once(override="B")
        print("model_used:", body["model_used"])

    print("\n=== Non-sticky canary estimate (200 calls) ===")
    counts = {"A": 0, "B": 0}
    for _ in range(200):
        body, _ = infer_once()
        counts[body["model_used"]] += 1
    total = counts["A"] + counts["B"]
    print("A:", counts["A"], "B:", counts["B"], f"(B% ≈ {100*counts['B']/total:.1f}%)")

    print("\n=== Sticky cookie check ===")
    with httpx.Client(timeout=10.0) as c:
        body, cookies = infer_once(client=c)
        g = body["model_used"]
        ok = True
        for _ in range(10):
            body2, _ = infer_once(client=c)
            ok = ok and (body2["model_used"] == g)
        print("group:", g, "sticky ok:", ok)

    print("\n=== Shadow delta ===")
    ab0, sh0 = grep_metrics()
    for _ in range(10):
        infer_once()  # normal calls trigger shadow if enabled
    ab1, sh1 = grep_metrics()
    print("AB assignments:\n", ab1)
    print("Shadow before:\n", sh0)
    print("Shadow after:\n", sh1)
