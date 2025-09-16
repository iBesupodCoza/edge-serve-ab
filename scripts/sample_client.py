#!/usr/bin/env python
import base64
import io

import numpy as np
import requests
from PIL import Image


def make_img_b64(size=224):
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    im = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":
    url = "http://127.0.0.1:8080/v1/infer"
    b64 = make_img_b64(224)
    r = requests.post(url, json={"image_b64": b64, "img_size": 224}, timeout=10)
    print(r.status_code, r.json())
