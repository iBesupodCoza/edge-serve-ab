from __future__ import annotations

import hashlib
import random
from typing import Literal, TypedDict

from fastapi import Request

Group = Literal["A", "B"]


class ABCfg(TypedDict, total=False):
    ab_weight_a: float
    ab_weight_b: float
    canary_enabled: bool
    shadow_enabled: bool
    sticky_cookie: str


def _det_hash(val: str) -> float:
    h = hashlib.md5(val.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big")
    return (v % 10_000_000) / 10_000_000.0


def choose_group(req: Request, cfg: ABCfg) -> Group:
    """Header override -> sticky cookie -> user_id hash -> weighted random."""
    # 1) Explicit override
    override = req.headers.get("X-Model-Override")
    if override == "A":
        return "A"
    if override == "B":
        return "B"

    # 2) Sticky cookie
    cookie_name = cfg.get("sticky_cookie", "ab_group")
    ck = req.cookies.get(cookie_name)
    if ck == "A":
        return "A"
    if ck == "B":
        return "B"

    # 3) Deterministic by user_id header (optional)
    user_id = req.headers.get("X-User-Id") or req.headers.get("user_id")
    w_b = float(cfg.get("ab_weight_b", 0.1)) if cfg.get("canary_enabled", True) else 0.0
    w_b = max(0.0, min(1.0, w_b))
    if user_id:
        r = _det_hash(user_id)
        return "B" if r < w_b else "A"

    # 4) Weighted random
    return "B" if random.random() < w_b else "A"


def other(group: Group) -> Group:
    return "B" if group == "A" else "A"
