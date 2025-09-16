from __future__ import annotations

import asyncio
import shutil
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.config import settings
from app.models.runtime import ONNXBatchInferencer, RuntimeConfig, pick_providers


def require_admin_token(req: Request) -> None:
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer")
    token = auth.split(" ", 1)[1]
    if token != settings.admin_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad token")


router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_token)])


class ConfigUpdate(BaseModel):
    ab_weight_a: float | None = Field(None, ge=0.0, le=1.0)
    ab_weight_b: float | None = Field(None, ge=0.0, le=1.0)
    canary_enabled: bool | None = None
    shadow_enabled: bool | None = None


@router.get("/config")
def get_cfg(req: Request) -> dict[str, Any]:
    return dict(req.app.state.ab_cfg)


@router.post("/config")
def set_cfg(update: ConfigUpdate, req: Request) -> dict[str, Any]:
    cfg = req.app.state.ab_cfg
    # Apply provided fields
    for k, v in update.model_dump(exclude_none=True).items():
        cfg[k] = v

    # Validate weights
    a = float(cfg.get("ab_weight_a", 0.9))
    b = float(cfg.get("ab_weight_b", 0.1))
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise HTTPException(400, "Weights must be in [0,1]")
    if abs((a + b) - 1.0) > 1e-6:
        raise HTTPException(400, "Weights must sum to 1.0")
    return dict(cfg)


@router.post("/warmup")
async def warmup(req: Request, runs: int = 3, img_size: int = 224) -> dict[str, Any]:
    await asyncio.gather(
        req.app.state.infer_a.warmup(img_size=img_size, runs=runs),
        req.app.state.infer_b.warmup(img_size=img_size, runs=runs),
    )
    return {"ok": True, "runs": runs, "img_size": img_size}


@router.post("/promote")
async def promote(req: Request) -> dict[str, Any]:
    """
    Blue/Green style: promote vB->vA.
    1) Copy vB.onnx over vA.onnx
    2) Recreate A session and warmup
    """
    vb = settings.model_vb_path
    va = settings.model_va_path
    shutil.copy2(vb, va)

    providers = pick_providers(settings.ort_providers)
    # Close old A and swap
    await req.app.state.infer_a.close()
    cfgA = RuntimeConfig(
        model_name="A",
        path=va,
        batch_max_size=settings.batch_max_size,
        batch_max_wait_s=settings.batch_max_wait_ms / 1000.0,
        queue_max=settings.queue_max,
    )
    req.app.state.infer_a = ONNXBatchInferencer(cfgA, providers=providers)
    await req.app.state.infer_a.warmup(img_size=224, runs=3)
    return {"ok": True, "promoted": "B->A"}
