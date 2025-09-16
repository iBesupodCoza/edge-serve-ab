# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# onnxruntime CPU runtime needs libgomp at minimum
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# If you already have a requirements.txt, this will use it.
# If not, use the one provided below in this answer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src ./src
# Copy models if you keep them in-repo
# (safe even if the folder doesn't exist; you can delete this line if you prefer)
COPY models ./models

# Reasonable defaults; override in compose/.env/K8s as needed
ENV MAX_BODY_BYTES=10485760 \
    RATE_LIMIT_RPS=5 \
    RATE_LIMIT_BURST=10 \
    AB_WEIGHT_A=1.0 \
    AB_WEIGHT_B=0.0 \
    CANARY_ENABLED=false \
    SHADOW_ENABLED=true \
    STICKY_COOKIE=ab_group

EXPOSE 8080

# FastAPI entrypoint
CMD ["uvicorn","--app-dir","src","app.server:create_app","--factory","--host","0.0.0.0","--port","8080"]
