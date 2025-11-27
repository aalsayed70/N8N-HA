# src/export_onnx.py
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from model import DaryWakeWordCNN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CHECKPOINT_PATH = "experiments/dary_cnn/checkpoints/best_model.pt"
ONNX_PATH = "experiments/dary_cnn/dary_wakeword.onnx"

def export_onnx():
    model = DaryWakeWordCNN()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    # Example input: batch=1, n_mels=40, time=100 (approx, dynamic)
    dummy_input = torch.randn(1, 1, 40, 100)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "time_steps"},
            "logits": {0: "batch_size"},
        },
        opset_version=13
    )
    print("Exported to", ONNX_PATH)

if __name__ == "__main__":
    export_onnx()
