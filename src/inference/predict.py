import time
from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort
from PIL import Image

# ---------------------------------------------------------------------------
# PlantVillage — 15-class subset (EfficientNet-B0 fine-tuned)
# Order must match the class index used during training
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Miner",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_mosaic_virus",
    "Tomato_healthy",
]

# ---------------------------------------------------------------------------
# Preprocessing constants — must match training transforms exactly
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet mean
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # ImageNet std
_INPUT_SIZE = 224


# ---------------------------------------------------------------------------
# Helper — preprocessing
# ---------------------------------------------------------------------------


def _preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize → CenterCrop → Normalise → NCHW float32 batch of 1.
    Mirrors torchvision.transforms used during training.
    """
    # 1. Resize shortest edge to 256, then centre-crop to 224
    w, h = image.size
    scale = 256 / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - _INPUT_SIZE) // 2
    top = (new_h - _INPUT_SIZE) // 2
    image = image.crop((left, top, left + _INPUT_SIZE, top + _INPUT_SIZE))

    # 2. Convert to float32 numpy array in [0, 1]
    arr = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)

    # 3. Normalise with ImageNet stats
    arr = (arr - _MEAN) / _STD  # (H, W, 3)

    # 4. HWC → CHW → NCHW (batch=1)
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 224, 224)

    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Helper — softmax (ONNX Runtime returns raw logits)
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Core inference engine
# ---------------------------------------------------------------------------


class PlantDiseaseClassifier:
    """
    ONNX Runtime inference engine for the EfficientNet-B0 plant disease model.

    Usage
    -----
    clf = PlantDiseaseClassifier("plant_disease_classifier.onnx")
    result = clf.predict("leaf.jpg")
    result = clf.predict(pil_image)
    result = clf.predict(numpy_hwc_uint8)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cpu",  # "cpu" | "cuda" | "tensorrt"
        intra_op_threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device.lower()

        providers = self._resolve_providers(self.device)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = intra_op_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=providers,
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 5,
    ) -> dict:
        """
        Run inference on a single image.

        Parameters
        ----------
        image : file path | PIL.Image | numpy uint8 HWC array
        top_k : number of top predictions to return

        Returns
        -------
        {
            "predicted_class": str,
            "confidence":      float,          # 0–1
            "is_healthy":      bool,
            "top_k": [
                {"class": str, "confidence": float},
                ...
            ],
            "latency_ms": float,
        }
        """
        pil_image = self._to_pil(image)
        tensor = _preprocess(pil_image)

        t0 = time.perf_counter()
        logits = self._session.run([self._output_name], {self._input_name: tensor})[
            0
        ]  # (1, num_classes)
        latency = (time.perf_counter() - t0) * 1000  # ms

        probs = _softmax(logits)[0]  # (num_classes,)
        top_k = min(top_k, len(CLASS_NAMES))
        top_idx = np.argsort(probs)[::-1][:top_k]

        predicted_idx = int(top_idx[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx])

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "is_healthy": "healthy" in predicted_class.lower(),
            "top_k": [
                {
                    "class": CLASS_NAMES[i],
                    "confidence": round(float(probs[i]), 4),
                }
                for i in top_idx
            ],
            "latency_ms": round(latency, 3),
        }

    def warmup(self, n: int = 3) -> None:
        """Run n dummy inferences to warm up the runtime."""
        dummy = np.random.rand(1, 3, _INPUT_SIZE, _INPUT_SIZE).astype(np.float32)
        for _ in range(n):
            self._session.run([self._output_name], {self._input_name: dummy})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_providers(device: str) -> list[str]:
        available = ort.get_available_providers()

        if device == "tensorrt":
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError(
                    "TensorRT execution provider not available. "
                    "Install onnxruntime-gpu with TensorRT support."
                )
            return [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",  # fallback for unsupported ops
                "CPUExecutionProvider",
            ]

        if device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDA execution provider not available. " "Install onnxruntime-gpu."
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

    @staticmethod
    def _to_pil(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")


# ---------------------------------------------------------------------------
# CLI — quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Plant disease inference")
    parser.add_argument("image", help="Path to leaf image")
    parser.add_argument(
        "--model",
        default="plant_disease_classifier.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "tensorrt"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    clf = PlantDiseaseClassifier(args.model, device=args.device)
    clf.warmup()

    result = clf.predict(args.image, top_k=args.top_k)

    print(json.dumps(result, indent=2))
