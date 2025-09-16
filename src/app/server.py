from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.admin import router as admin_router
from .api.health import router as health_router
from .api.v1 import router as v1_router
from .config import settings
from .limits import RateLimiter
from .middleware.trace import TraceIdMiddleware  # ✅ correct import
from .models.runtime import ONNXBatchInferencer, RuntimeConfig, pick_providers
from .obs.logging import setup_logging
from .obs.metrics import MetricsMiddleware, metrics_router
from .obs.payload import PayloadLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.models_loaded = False

    providers = pick_providers(settings.ort_providers)

    cfgA = RuntimeConfig(
        model_name="A",
        path=settings.model_va_path,
        batch_max_size=settings.batch_max_size,
        batch_max_wait_s=settings.batch_max_wait_ms / 1000.0,
        queue_max=settings.queue_max,
    )
    cfgB = RuntimeConfig(
        model_name="B",
        path=settings.model_vb_path,
        batch_max_size=settings.batch_max_size,
        batch_max_wait_s=settings.batch_max_wait_ms / 1000.0,
        queue_max=settings.queue_max,
    )

    app.state.infer_a = ONNXBatchInferencer(cfgA, providers=providers)
    app.state.infer_b = ONNXBatchInferencer(cfgB, providers=providers)

    await asyncio.gather(
        app.state.infer_a.warmup(img_size=224, runs=3),
        app.state.infer_b.warmup(img_size=224, runs=3),
    )

    # --- A/B config (mutable) with env overrides & normalization ---
    def _read_float(env_key: str, default_val: float) -> float:
        raw = os.getenv(env_key)
        if raw is None:
            return float(default_val)
        try:
            return float(raw)
        except ValueError:
            return float(default_val)

    a_raw = _read_float("AB_WEIGHT_A", getattr(settings, "ab_weight_a", 1.0))
    b_raw = _read_float("AB_WEIGHT_B", getattr(settings, "ab_weight_b", 0.0))
    if a_raw < 0 or b_raw < 0:
        a_raw, b_raw = 1.0, 0.0
    total = a_raw + b_raw
    if total <= 0:
        ab_weight_a, ab_weight_b = 1.0, 0.0
    else:
        ab_weight_a, ab_weight_b = a_raw / total, b_raw / total

    app.state.ab_cfg = {
        "ab_weight_a": ab_weight_a,
        "ab_weight_b": ab_weight_b,
        "canary_enabled": settings.canary_enabled,
        "shadow_enabled": settings.shadow_enabled,
        "sticky_cookie": settings.sticky_cookie,
    }

    app.state.models_loaded = True
    try:
        yield
    finally:
        await asyncio.gather(
            app.state.infer_a.close(),
            app.state.infer_b.close(),
            return_exceptions=True,
        )


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- Read env at app creation (so tests with monkeypatch take effect) ---
    max_body_bytes = int(os.getenv("MAX_BODY_BYTES", str(settings.max_body_bytes)))
    rl_rate = float(os.getenv("RATE_LIMIT_RPS", str(settings.rate_limit_rps)))
    rl_burst = float(os.getenv("RATE_LIMIT_BURST", str(settings.rate_limit_burst)))

    # Store a live limiter instance on app.state (used by a dependency)
    app.state.rate_limiter = RateLimiter(rate=rl_rate, burst=rl_burst)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=[],
    )

    # Metrics
    app.add_middleware(MetricsMiddleware)

    # Payload size guard
    app.add_middleware(PayloadLimitMiddleware, max_body_bytes=max_body_bytes)

    # Tracing (add LAST so it wraps everything and runs FIRST)
    app.add_middleware(TraceIdMiddleware)

    # Routers
    app.include_router(metrics_router)
    app.include_router(health_router)
    app.include_router(v1_router)  # /v1 endpoints (rate-limited via dependency)
    app.include_router(admin_router)

    return app
