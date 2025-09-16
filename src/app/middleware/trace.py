from __future__ import annotations

import uuid
from collections.abc import Iterable

from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _decode_headers(headers: Iterable[tuple[bytes, bytes]]) -> dict[str, str]:
    return {k.decode("latin1").lower(): v.decode("latin1") for k, v in headers}


def _extract_from_traceparent(val: str) -> str | None:
    # W3C traceparent: version-traceid-spanid-flags
    parts = val.split("-")
    if len(parts) >= 3 and len(parts[1]) in (16, 32):
        return parts[1]
    return None


def _pick_trace_id(hdrs: dict[str, str]) -> str:
    if v := hdrs.get("x-request-id", "").strip():
        return v
    if v := hdrs.get("trace-id", "").strip():
        return v
    if v := hdrs.get("x-correlation-id", "").strip():
        return v
    if "traceparent" in hdrs:
        maybe = _extract_from_traceparent(hdrs["traceparent"])
        if maybe:
            return maybe
    return uuid.uuid4().hex


class TraceIdMiddleware:
    """
    Ensures every HTTP response has a stable request id:
      - "Trace-Id" and "X-Request-ID"
    """

    def __init__(self, app: ASGIApp, header_name: str = "Trace-Id") -> None:
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        hdrs = _decode_headers(scope.get("headers", []))
        trace_id = _pick_trace_id(hdrs)
        scope["trace_id"] = trace_id

        async def send_with_trace(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((self.header_name.encode("latin1"), trace_id.encode("latin1")))
                headers.append((b"x-request-id", trace_id.encode("latin1")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_trace)
