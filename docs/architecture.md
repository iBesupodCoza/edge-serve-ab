# edge-serve-ab — Architecture

## Overview
`edge-serve-ab` is a FastAPI service that serves ONNX image classifiers with:
- **A/B routing** between Model A and Model B (sticky cookie support)
- **Shadow traffic** to cross-check the “other” model
- **Admin endpoints** to tweak routing and promote B→A
- **Observability**: Prometheus metrics, structured logs, request trace IDs
- **Guards**: payload size limits + token-bucket rate limiting

---

HTTP → ASGI Middlewares

* TraceIdMiddleware (adds/propagates Trace-Id header)
* MetricsMiddleware (request counters/histograms)
* PayloadLimitMiddleware (413 on large bodies)
  ↓
  FastAPI Routers
* /healthz /health /ready (liveness/readiness)
* /metrics (Prometheus)
* /v1/infer (inference, A/B + optional shadow)
* /admin/\* (runtime config + warmup + promote)
  ↓
  ONNXBatchInferencer(A + B)
* batching & queueing
* ONNX Runtime providers (CPU by default)

---

## A/B & Shadow
- Assignment uses weights `ab_weight_a`/`ab_weight_b` from `app.state.ab_cfg`.
- A **sticky cookie** (`ab_group`) keeps users pinned to A or B.
- If shadow enabled, the non-primary model gets a *non-blocking* short-deadline request.

---

## Rate Limiting & Payload Guard
- **Token bucket** per-client: `RATE_LIMIT_RPS` / `RATE_LIMIT_BURST`.
- **Payload limit**: `MAX_BODY_BYTES` (413 if exceeded).

---

## Observability
- **Metrics**: `/metrics` (Prometheus). Example metrics:
  - request counts/latencies, rate-limit count, payload rejects, A/B assignment, shadow success/error.
- **Logs**: `structlog` JSON.
- **Trace IDs**: Header `Trace-Id` on every response (respects incoming `X-Request-ID` if present).

---

## Admin Flow
1. `POST /admin/config` to adjust AB/Shadow flags and weights.
2. `POST /admin/warmup` to pre-run inference.
3. `POST /admin/promote` performs **Blue/Green**: copies `vB.onnx` over `vA.onnx`, reloads session A, and warms it up.

---

## Deployment
- **Docker**: single container (see `Dockerfile` + `compose.yml`).
- **Kubernetes**: see `deploy/k8s/edge-serve-ab.yaml` (Deployment, Service, HPA, ConfigMap, Prometheus annotations).
