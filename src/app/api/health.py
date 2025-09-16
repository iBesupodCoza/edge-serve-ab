from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """
    Liveness: process is up and can serve requests.
    Unit test expects: {"status": "ok"}.
    """
    return {"status": "ok"}


@router.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, bool]:
    """
    Kubernetes-style liveness endpoint.
    Integration test expects: {"ok": True}.
    """
    return {"ok": True}


@router.get("/ready")
def ready(req: Request) -> JSONResponse:
    """
    Readiness: models loaded & warmed up.
    Unit test expects keys: {"ready": bool, "models_loaded": bool}.
    """
    models_loaded: bool = bool(getattr(req.app.state, "models_loaded", False))
    body = {"ready": models_loaded, "models_loaded": models_loaded}
    return JSONResponse(body, status_code=200 if models_loaded else 503)


@router.get("/readyz", include_in_schema=False)
def readyz(req: Request) -> JSONResponse:
    """
    Kubernetes-style readiness endpoint.
    Mirrors /ready.
    """
    return ready(req)
