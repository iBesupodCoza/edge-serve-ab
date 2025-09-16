from pathlib import Path

import onnx


def test_models_exist_and_valid():
    mdir = Path("models")
    va = mdir / "vA.onnx"
    vb = mdir / "vB.onnx"
    assert va.exists(), "vA.onnx missing"
    assert vb.exists(), "vB.onnx missing"
    onnx.checker.check_model(onnx.load(str(va)))
    onnx.checker.check_model(onnx.load(str(vb)))
