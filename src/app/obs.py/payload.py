from __future__ import annotations

from fastapi.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.obs.metrics import PAYLOAD_REJECTS

from ..config import settings


def _get_content_length(scope: Scope) -> int | None:
    hdrs = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
    if "content-length" in hdrs:
        try:
            return int(hdrs["content-length"])
        except ValueError:
            return None
    return None


class PayloadLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int | None = None) -> None:
        self.app = app
        # Use value passed from server.create_app(), fallback to settings
        cfg_val = max_body_bytes if max_body_bytes is not None else settings.max_body_bytes
        self.max_body_bytes = int(cfg_val) if cfg_val is not None else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method", "GET") in ("POST", "PUT", "PATCH"):
            clen = _get_content_length(scope)
            if self.max_body_bytes is not None and clen is not None and clen > self.max_body_bytes:
                PAYLOAD_REJECTS.labels(route=scope.get("path", "unknown")).inc()
                resp = PlainTextResponse("Payload too large", status_code=413)
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)
