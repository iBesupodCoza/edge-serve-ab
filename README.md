# edge-serve-ab

A fast A/B image-inference service with production-grade guardrails and observability:
- **A/B routing** with sticky cookies and optional **shadow traffic**
- **FastAPI** + **ONNX Runtime** (batching)
- **Safety**: request **rate limiting** + **payload size** limits
- **Observability**: **Prometheus** metrics, **Grafana** dashboard, JSON logs
- **Traceability**: every response has a **`Trace-Id`** (respects incoming `X-Request-ID`)
- **Admin** controls: live tweak A/B weights, warmup, and **Blue/Green** promote B→A
- Turn-key **Docker** / **Compose** / **Kubernetes** deployment

[![CI](https://github.com/abdulvahapmutlu/edge-serve-ab/actions/workflows/ci.yml/badge.svg)](https://github.com/abdulvahapmutlu/edge-serve-ab/actions/workflows/ci.yml)
[![Docker](https://github.com/abdulvahapmutlu/edge-serve-ab/actions/workflows/docker.yml/badge.svg)](https://github.com/abdulvahapmutlu/edge-serve-ab/actions/workflows/docker.yml)

---

## Quickstart (Local)

**Python**
```
python -m venv .venv
source .venv/bin/activate            # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn --app-dir src app.server:create_app --factory --host 0.0.0.0 --port 8080
````

**Docker**

```
docker build -t edge-serve-ab:local .
docker run --rm -p 8080:8080 --name edge-serve edge-serve-ab:local
```

**Local stack (App + Prometheus + Grafana)**

```
docker compose up -d
# App:        http://localhost:8080
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (user/pass: admin/admin by default)
```

---

## Configuration

Copy `.env.example` → `.env` and adjust as needed. Key settings:

* **Traffic & safety**

  * `MAX_BODY_BYTES` – reject large payloads (413)
  * `RATE_LIMIT_RPS`, `RATE_LIMIT_BURST` – token-bucket rate limiter
* **A/B config**

  * `AB_WEIGHT_A`, `AB_WEIGHT_B` – routing weights
  * `SHADOW_ENABLED`, `CANARY_ENABLED`, `STICKY_COOKIE`
* **Admin**

  * `ADMIN_TOKEN` – required for `/admin/*`
* **Models**

  * `MODEL_VA_PATH`, `MODEL_VB_PATH` – ONNX model paths (defaults to `models/vA.onnx`, `models/vB.onnx`)

---

## API

### Health

* `GET /healthz` → `{"ok": true}`
* `GET /health` → `{"status": "ok"}`
* `GET /ready` → `{"ready": true, "models_loaded": true}`

### Inference

`POST /v1/infer`
Request:

```
{
  "image_b64": "<base64 png or jpeg>",
  "img_size": 224
}
```

Response:

```
{
  "trace_id": "uuid",
  "model_used": "A",
  "top5": [[123, 0.25], [42, 0.20], [7, 0.15], [99, 0.11], [256, 0.05]],
  "shape": [1000]
}
```

**Headers:**

* Server returns `Trace-Id` on every response.
* If the request includes `X-Request-ID`, it will be used as `Trace-Id`.

### Admin (Bearer token)

```
# Get current config
curl -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8080/admin/config

# Update A/B weights & toggles
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
     -d '{"ab_weight_a":0.9,"ab_weight_b":0.1,"shadow_enabled":true}' \
     http://localhost:8080/admin/config

# Warm up both models
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8080/admin/warmup

# Blue/Green promote B→A
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8080/admin/promote
```

---

## Observability

* **Prometheus** metrics at `/metrics` (scraped by Prometheus).
* **Grafana** dashboard included: `grafana/dashboards/edge-serve-ab.json`.
* Local provisioning for Grafana and Prometheus is under `local/`.

Example metrics:

* Request counts & latency histograms
* Rate-limit events
* Payload rejects
* A/B assignment
* Shadow success/error

---

## Load Testing

k6 scripts live under `load/k6/`.

**Health load**

```
docker run --rm -i --network host grafana/k6 run - < load/k6/healthz.js
```

**Inference load**

```
docker run --rm -i --network host grafana/k6 run - < load/k6/infer.js
```

Export results to `results/` and visualize as needed.

---

## Development

Install dev tools and run checks:

```
pip install -r requirements.txt
pip install black ruff mypy pytest

ruff check .
black --check .
mypy src
pytest -q
```

Optionally enable pre-commit:

```
pre-commit install
```

---

## CI/CD

This repo includes GitHub Actions:

* **CI** (`.github/workflows/ci.yml`): lint, type-check, test.
* **Docker** (`.github/workflows/docker.yml`): build & push image to GHCR (`ghcr.io/<owner>/edge-serve-ab`).

---

## Deploy (Kubernetes)

Single manifest:

```
kubectl apply -f deploy/k8s/edge-serve-ab.yaml
kubectl -n edge-serve-ab get pods
```

---

## License

This project is licensed under Apache 2.0 license
