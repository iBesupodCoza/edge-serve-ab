from __future__ import annotations

import asyncio
import base64
import io
import uuid

import numpy as np
import structlog
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from PIL import Image
from pydantic import BaseModel, Field

from app.config import settings
from app.models.router import choose_group
from app.obs.metrics import AB_ASSIGN, SHADOW_REQS

from ..limits import enforce_rate_limit

router = APIRouter(prefix="/v1", tags=["v1"])
log = structlog.get_logger()

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InferRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded RGB image (PNG/JPEG).")
    img_size: int = Field(224, ge=64, le=640, description="Resize to this square size")


class InferResponse(BaseModel):
    trace_id: str
    model_used: str
    top5: list[tuple[int, float]]
    shape: tuple[int, ...]


def _decode_and_preprocess(b64s: str, img_size: int) -> np.ndarray:
    raw = base64.b64decode(b64s)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    # Resize shortest side -> 256; center crop -> img_size
    w, h = img.size
    if w <= h:
        new_w, new_h = 256, int(h * (256.0 / w))
    else:
        new_h, new_w = 256, int(w * (256.0 / h))
    img = img.resize((new_w, new_h))
    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))

    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def _top5(logits: np.ndarray) -> list[tuple[int, float]]:
    x = logits.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    p = e / (np.sum(e) + 1e-12)
    idx = np.argsort(p)[-5:][::-1]
    return [(int(i), float(p[i])) for i in idx]


async def _fire_shadow(app: FastAPI, model_from: str, x: np.ndarray, deadline: float) -> None:
    """Non-blocking shadow call to the other model; swallow all errors."""
    try:
        target = "B" if model_from == "A" else "A"
        inf = app.state.infer_b if target == "B" else app.state.infer_a
        # Shorten deadline for shadow so it never competes with primary
        shadow_deadline = min(deadline, asyncio.get_running_loop().time() + 0.05)
        _ = await inf.infer(x, deadline=shadow_deadline)
        SHADOW_REQS.labels(model_from, target, "ok").inc()
    except Exception as e:
        SHADOW_REQS.labels(model_from, "B" if model_from == "A" else "A", "err").inc()
        log.warning("shadow_error", from_model=model_from, err=str(e))


@router.post("/infer", response_model=InferResponse, dependencies=[Depends(enforce_rate_limit)])
async def infer(req: Request, body: InferRequest, response: Response) -> InferResponse:
    app = req.app
    # Deadline for this request
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (settings.req_timeout_ms / 1000.0)

    # Decode
    x = _decode_and_preprocess(body.image_b64, body.img_size)

    # Decide A/B
    cfg = app.state.ab_cfg
    group = choose_group(req, cfg)
    AB_ASSIGN.labels(group).inc()

    # Sticky cookie set if missing
    cookie_name = cfg.get("sticky_cookie", "ab_group")
    if cookie_name not in req.cookies:
        # 7 days sticky
        response.set_cookie(
            cookie_name,
            group,
            max_age=7 * 24 * 3600,
            httponly=False,
            samesite="lax",
        )

    # Select primary inferencer
    primary = app.state.infer_a if group == "A" else app.state.infer_b
    out = await primary.infer(x, deadline=deadline)

    # Optional shadow traffic to the other model (non-blocking)
    sh_enabled = bool(cfg.get("shadow_enabled", True))
    override = req.headers.get("X-Model-Override")
    shadow_forced = override == "shadow"
    if sh_enabled or shadow_forced:
        # Do not await; fire and forget — keep a reference (satisfies RUF006)
        task = asyncio.create_task(_fire_shadow(app, group, x, deadline))
        if not hasattr(app.state, "bg_tasks"):
            app.state.bg_tasks = set()
        app.state.bg_tasks.add(task)
        task.add_done_callback(app.state.bg_tasks.discard)

    return InferResponse(
        trace_id=str(uuid.uuid4()),
        model_used=group,
        top5=_top5(out),
        shape=tuple(out.shape),
    )
