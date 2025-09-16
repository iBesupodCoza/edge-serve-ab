#!/usr/bin/env python
"""
Export MobileNetV2 to ONNX (FP32) as vA.onnx and create an optimized candidate vB:
 - Option A (default): INT8 PTQ via ONNX Runtime quantization with synthetic calibration
 - Option B: if --no-int8, just copy FP32 and rename to vB (still useful for A/B router wiring)

Usage:
  python scripts/export_onnx.py --out-dir models --img-size 224 --opset 17
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# Optional: ONNX Runtime quantization (static)
try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    ORT_QUANT_AVAILABLE = True
except Exception:
    ORT_QUANT_AVAILABLE = False


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_pil(img: Image.Image, img_size: int) -> np.ndarray:
    # Resize shortest side to 256, center-crop to img_size, normalize to CHW float32
    img = torchvision.transforms.functional.resize(img, 256, antialias=True)
    img = torchvision.transforms.functional.center_crop(img, [img_size, img_size])
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC in [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def synthetic_batch(n: int, img_size: int) -> np.ndarray:
    # Normalized "image-like" noise around ImageNet mean
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(n, 3, img_size, img_size)).astype(np.float32)
    # Keep variance in a realistic range
    return x


def export_fp32(out_path: Path, img_size: int, opset: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights).eval()

    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = {"input": {0: "N"}, "logits": {0: "N"}}

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )

    # Sanity check
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)


class SyntheticCalibrationReader(CalibrationDataReader):
    """
    Minimal calibration data reader: yields a few batches of synthetic,
    normalized tensors that match the model's input name/shape.
    """

    def __init__(
        self,
        model_path: str,
        input_name: str,
        img_size: int,
        num_batches: int = 8,
        batch_size: int = 8,
    ):
        self.input_name = input_name
        self.img_size = img_size
        self.num_batches = num_batches
        self.batch_size = batch_size
        self._iter = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter >= self.num_batches:
            return None
        self._iter += 1
        data = synthetic_batch(self.batch_size, self.img_size)
        return {self.input_name: data}


def make_int8(fp32_path: Path, int8_path: Path, img_size: int) -> None:
    if not ORT_QUANT_AVAILABLE:
        raise RuntimeError(
            "ONNX Runtime quantization not available. Install onnxruntime>=1.15 with quantization extras."
        )

    # We know we exported with input name "input"
    data_reader = SyntheticCalibrationReader(str(fp32_path), input_name="input", img_size=img_size)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

    onnx.checker.check_model(onnx.load(str(int8_path)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("models"))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-int8", action="store_true", help="Skip int8 PTQ; copy FP32 to vB instead.")
    args = p.parse_args()

    out_dir = args.out_dir
    va_fp32 = out_dir / "vA.onnx"
    vb_int8 = out_dir / "vB.onnx"

    t0 = time.time()
    print(f"Exporting FP32 → {va_fp32}")
    export_fp32(va_fp32, img_size=args.img_size, opset=args.opset)
    print(f"✓ vA.onnx ready in {time.time() - t0:.2f}s")

    if args.no_int8:
        print("Skipping INT8; copying vA→vB for now")
        shutil.copy2(va_fp32, vb_int8)
    else:
        try:
            t1 = time.time()
            print(f"Quantizing to INT8 (QDQ) → {vb_int8}")
            make_int8(va_fp32, vb_int8, img_size=args.img_size)
            print(f"✓ vB.onnx (INT8) ready in {time.time() - t1:.2f}s")
        except Exception as e:
            print(f"[warn] INT8 quantization failed: {e}")
            print("       Falling back to copying FP32 to vB.")
            shutil.copy2(va_fp32, vb_int8)

    print("All done.")
    print(f"Artifacts:\n  - {va_fp32}\n  - {vb_int8}")


if __name__ == "__main__":
    main()
