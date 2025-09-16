import time

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Prometheus primitives
REQS = Counter("requests_total", "Total HTTP requests", ["route", "code", "method"])
LAT = Histogram(
    "latency_seconds",
    "Request latency (seconds)",
    ["route", "method"],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)
UP = Gauge("app_up", "Application up (1)")

metrics_router = APIRouter()


@metrics_router.get("/metrics")
def metrics() -> Response:
    content = generate_latest()
    return Response(content=content, media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware:
    """ASGI middleware to record request counts and latency by route & method."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        UP.set(1)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        method = scope.get("method", "GET")
        route_path: str | None = None  # set later when we have route

        async def send_wrapper(message: Message) -> None:
            nonlocal route_path
            if message["type"] == "http.response.start":
                status = message.get("status", 0)
                # Try to get a bound route from scope (falls back to raw path)
                route = scope.get("route")
                route_path = getattr(route, "path", scope.get("path", "unknown"))
                REQS.labels(route=route_path, code=str(status), method=method).inc()
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route_path = route_path or scope.get("path", "unknown")
            LAT.labels(route=route_path, method=method).observe(time.perf_counter() - start)


# Per-model metrics used by runtime
QDEP = Gauge("queue_depth", "Pending requests in model queue", ["model"])
BATCH_LAST = Gauge("batch_size_last", "Last executed batch size", ["model"])
INFER_LAT = Histogram(
    "inference_latency_seconds",
    "Model inference latency (seconds)",
    ["model"],
    buckets=(0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
)

# New: enforcement metrics
RATE_LIMITED = Counter("rate_limited_total", "Requests rejected by rate limiter", ["route"])
PAYLOAD_REJECTS = Counter(
    "payload_rejected_total", "Requests rejected due to payload size", ["route"]
)
CIRCUIT_OPEN = Gauge("circuit_open", "Circuit breaker open (1=open)", ["model"])

AB_ASSIGN = Counter("ab_assignments_total", "A/B assignments", ["group"])
SHADOW_REQS = Counter("shadow_requests_total", "Shadow requests fired", ["from", "to", "result"])
