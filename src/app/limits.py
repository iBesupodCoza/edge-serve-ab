from __future__ import annotations

import time

from fastapi import HTTPException, Request

from app.obs.metrics import RATE_LIMITED


class TokenBucket:
    def __init__(self, rate: float, burst: float) -> None:
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last = time.monotonic()

    def allow(self, n: float = 1.0) -> bool:
        now = time.monotonic()
        # refill
        self.tokens = min(self.burst, self.tokens + self.rate * (now - self.last))
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class RateLimiter:
    def __init__(self, rate: float, burst: float) -> None:
        self.rate = float(rate)
        self.burst = float(burst)
        self._buckets: dict[str, TokenBucket] = {}

    def _bucket(self, key: str) -> TokenBucket:
        b = self._buckets.get(key)
        if b is None:
            b = self._buckets[key] = TokenBucket(self.rate, self.burst)
        return b

    def check(self, key: str, route: str) -> None:
        if not self._bucket(key).allow(1.0):
            RATE_LIMITED.labels(route=route).inc()
            raise HTTPException(status_code=429, detail="Too Many Requests")


def client_key(req: Request) -> str:
    # Prefer real client if behind proxy you can use X-Forwarded-For
    return req.client.host if req.client else "unknown"


async def rate_limit_dep(req: Request) -> None:
    rl: RateLimiter = req.app.state.rate_limiter
    rl.check(client_key(req), route=req.url.path)


async def enforce_rate_limit(request: Request) -> None:
    """
    Dependency that uses the runtime-configured limiter stored on app.state.
    This lets tests tweak RATE_LIMIT_* via env and have it take effect.
    """
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    limiter.check(client_key(request), route=request.url.path)
