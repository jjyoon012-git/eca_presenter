#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from pathlib import Path
from train_eca_gesture import ECAGestureNet, NUM_CLASSES, ROOT_DIR

def export_to_onnx(
    weights_path: Path = ROOT_DIR / "model" / "eca_gesture.pth",
    onnx_path: Path = ROOT_DIR / "models" / "gesture_eca.onnx"
):
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = ECAGestureNet(num_classes=NUM_CLASSES)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=18
    )
    print(f"ONNX 모델을 {onnx_path}에 저장했습니다.")

if __name__ == "__main__":
    export_to_onnx()
