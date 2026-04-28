import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from model import build_model

# ── Config ────────────────────────────────────────────────────────────────
CHECKPOINT = "best_model.pth"
ONNX_PATH = "plant_disease_classifier.onnx"
NUM_CLASSES = 15
IMG_SIZE = 224
DEVICE = torch.device("cpu")  # Export from CPU for portability


# ── Load trained model ────────────────────────────────────────────────────
def load_model(checkpoint_path: str) -> torch.nn.Module:
    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=False).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}")
    return model


# ── Export ────────────────────────────────────────────────────────────────
def export_onnx(model: torch.nn.Module, path: str):
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)  # batch=1

    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,  # fuse constant ops → smaller model
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={  # allow variable batch size
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"Exported → {path}")


# ── Validate ONNX graph ───────────────────────────────────────────────────
def validate_onnx(path: str):
    model_onnx = onnx.load(path)
    onnx.checker.check_model(model_onnx)
    print("ONNX graph check passed ✅")


# ── Verify outputs match PyTorch ──────────────────────────────────────────
def verify_parity(model: torch.nn.Module, path: str):
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_out = session.run(["logits"], {"image": dummy.numpy()})[0]

    max_diff = np.abs(torch_out - ort_out).max()
    print(f"Max output difference (PyTorch vs ONNX): {max_diff:.2e}")
    assert max_diff < 1e-4, "Parity check failed — outputs diverge too much"
    print("Parity check passed ✅")


# ── Benchmark inference speed ─────────────────────────────────────────────
def benchmark(path: str, n_runs: int = 200):
    import time

    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    # Warmup
    for _ in range(10):
        session.run(["logits"], {"image": dummy})

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(["logits"], {"image": dummy})
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"Avg inference latency (CPU, batch=1): {elapsed:.2f} ms")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model(CHECKPOINT)

    export_onnx(model, ONNX_PATH)
    validate_onnx(ONNX_PATH)
    verify_parity(model, ONNX_PATH)
    benchmark(ONNX_PATH)

    print("\nExport complete ✅")
    print(f"Deploy: {ONNX_PATH}")
