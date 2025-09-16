#!/usr/bin/env python
"""
Run both vA (FP32) and vB (INT8 if available) with ONNX Runtime and report shape + latency.

Usage:
  python scripts/validate_onnx.py --models-dir models --batch 1 --runs 50
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def pick_providers() -> list[str]:
    # Try CUDA first if available, fallback to CPU
    eps = []
    try:
        if "CUDAExecutionProvider" in ort.get_available_providers():
            eps.append("CUDAExecutionProvider")
    except Exception:
        pass
    eps.append("CPUExecutionProvider")
    return eps


def run_latency(
    sess: ort.InferenceSession, x: np.ndarray, runs: int = 50, warmup: int = 5
) -> dict[str, float]:
    input_name = sess.get_inputs()[0].name
    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        times.append(time.perf_counter() - t0)
    return {
        "p50_ms": 1000.0 * statistics.median(times),
        "p95_ms": 1000.0 * np.percentile(times, 95),
        "mean_ms": 1000.0 * statistics.mean(times),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    va = args.models_dir / "vA.onnx"
    vb = args.models_dir / "vB.onnx"
    assert va.exists() and vb.exists(), "Export models first."

    # Basic structural checks
    onnx.checker.check_model(onnx.load(str(va)))
    onnx.checker.check_model(onnx.load(str(vb)))

    providers = pick_providers()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"Using providers: {providers}")
    sess_a = ort.InferenceSession(str(va), sess_options=so, providers=providers)
    sess_b = ort.InferenceSession(str(vb), sess_options=so, providers=providers)

    N, C, H, W = args.batch, 3, args.img_size, args.img_size
    x = np.random.randn(N, C, H, W).astype(np.float32)

    out_a = sess_a.run(None, {sess_a.get_inputs()[0].name: x})[0]
    out_b = sess_b.run(None, {sess_b.get_inputs()[0].name: x})[0]
    print(f"Shapes: A={out_a.shape}, B={out_b.shape}")

    la = run_latency(sess_a, x, runs=args.runs)
    lb = run_latency(sess_b, x, runs=args.runs)
    print(f"Latency A (FP32): {la}")
    print(f"Latency B (candidate): {lb}")
    print("Done.")


if __name__ == "__main__":
    main()
