from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import onnxruntime as ort

from app.obs.metrics import BATCH_LAST, CIRCUIT_OPEN, INFER_LAT, QDEP


@dataclass
class RuntimeConfig:
    model_name: str
    path: str
    batch_max_size: int
    batch_max_wait_s: float
    queue_max: int
    cb_fail_threshold: int = 5
    cb_reset_after_s: float = 30.0


def pick_providers(pref: str = "AUTO") -> list[str]:
    # AUTO = try CUDA then CPU; CPU = force CPU
    pref = (pref or "AUTO").upper()
    providers: list[str] = []
    if pref == "AUTO":
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
        except Exception:
            pass
    providers.append("CPUExecutionProvider")
    return providers


class ONNXBatchInferencer:
    """
    Bounded-queue async batcher around an ONNX Runtime session.
    - Accepts single-item requests via `infer(x, deadline)`.
    - Packs into batches up to `batch_max_size` or `batch_max_wait_s`.
    - Returns 429 if queue is full, 503 on deadline overrun.
    """

    def __init__(self, cfg: RuntimeConfig, providers: list[str]) -> None:
        self.cfg = cfg
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(cfg.path, sess_options=so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

        self.q: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue(
            maxsize=cfg.queue_max
        )
        self._worker = asyncio.create_task(self._loop())
        self.cb_failures = 0
        self.cb_open_until = 0.0

    async def close(self) -> None:
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass

    async def warmup(self, img_size: int = 224, runs: int = 3) -> None:
        x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        for _ in range(runs):
            self.sess.run(None, {self.input_name: x})

    async def infer(self, x: np.ndarray, deadline: float) -> np.ndarray:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        try:
            self.q.put_nowait((x, fut))
            QDEP.labels(self.cfg.model_name).set(self.q.qsize())
        except asyncio.QueueFull:
            # Fail fast under saturation
            raise_queue_full()
        # Wait until the worker sets result or we hit deadline
        timeout = max(0.0, deadline - loop.time())
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError as _:
            # Inform the worker that we gave up; but worker will set the result regardless.
            raise_deadline()

    async def _loop(self) -> None:
        while True:
            xs: list[np.ndarray] = []
            futs: list[asyncio.Future] = []

            # Block for the first item
            x, f = await self.q.get()
            xs.append(x)
            futs.append(f)
            start = asyncio.get_running_loop().time()

            # Drain quickly to form a batch
            while len(xs) < self.cfg.batch_max_size:
                # Small time budget to accumulate
                if (asyncio.get_running_loop().time() - start) >= self.cfg.batch_max_wait_s:
                    break
                try:
                    x, f = self.q.get_nowait()
                    xs.append(x)
                    futs.append(f)
                except asyncio.QueueEmpty:
                    # Yield briefly to let other tasks post
                    await asyncio.sleep(0.0005)

            BATCH_LAST.labels(self.cfg.model_name).set(len(xs))
            QDEP.labels(self.cfg.model_name).set(self.q.qsize())

            batch = np.stack(xs, axis=0).astype(np.float32)
            t0 = time.perf_counter()
            try:
                out = self.sess.run(None, {self.input_name: batch})[0]
                INFER_LAT.labels(self.cfg.model_name).observe(time.perf_counter() - t0)
                # Success: reset breaker
                self.cb_failures = 0
                CIRCUIT_OPEN.labels(self.cfg.model_name).set(0)
                # Fan-out
                for i, fut in enumerate(futs):
                    if not fut.done():
                        fut.set_result(out[i])
            except Exception:
                # Failure: increment breaker, maybe open
                self.cb_failures += 1
                if self.cb_failures >= self.cfg.cb_fail_threshold:
                    self.cb_open_until = (
                        asyncio.get_running_loop().time() + self.cfg.cb_reset_after_s
                    )
                    CIRCUIT_OPEN.labels(self.cfg.model_name).set(1)
                # Propagate 503 to all waiting callers
                from fastapi import HTTPException

                ex = HTTPException(status_code=503, detail="Model execution failed")
                for fut in futs:
                    if not fut.done():
                        fut.set_exception(ex)


def raise_queue_full() -> NoReturn:
    from fastapi import HTTPException

    raise HTTPException(status_code=429, detail="Queue full")


def raise_circuit() -> NoReturn:
    from fastapi import HTTPException

    raise HTTPException(status_code=503, detail="Circuit open")


def raise_deadline() -> NoReturn:
    from fastapi import HTTPException

    raise HTTPException(status_code=503, detail="Timed out")
