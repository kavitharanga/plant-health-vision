# 🌿 Plant Disease Classifier

A deep learning pipeline that detects and classifies plant diseases from images. Built with **PyTorch**, exportable to **ONNX**, and served via a REST **API** — with both a CLI and notebook for quick inference.

---

## 🚀 Features

- 🧠 **Custom model** defined in [`src/model.py`](src/model.py)
- 📦 **Dataset loading** via [`src/dataset.py`](src/dataset.py)
- 🏋️ **Training** with [`src/train.py`](src/train.py)
- 📊 **Evaluation** with [`src/evaluate.py`](src/evaluate.py)
- 📤 **ONNX export** via [`src/onnx_export.py`](src/onnx_export.py)
- 🔍 **CLI inference** via [`src/inference/predict.py`](src/inference/predict.py)
- 🌐 **REST API** via [`src/api/main.py`](src/api/main.py)
- 📓 **Notebook** for experimentation: [`src/test.ipynb`](src/test.ipynb)

---

## 📁 Project Structure

```
src/
├── model.py              # Model architecture
├── dataset.py            # Dataset & data loading
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── onnx_export.py        # Export model to ONNX
├── test.ipynb            # Jupyter notebook
├── api/
│   ├── main.py           # FastAPI app
│   └── schemas.py        # Request/response schemas
└── inference/
    └── predict.py        # Prediction helper (CLI)

best_model.pth                    # Saved PyTorch weights
plant_disease_classifier.onnx     # Exported ONNX model
run.py                            # Convenience entrypoint
```

---

## ⚙️ Setup

### Option A — Poetry

```sh
poetry install
poetry shell
```

### Option B — pip

```sh
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🏋️ Train

```sh
python -m src.train
```

The best model weights will be saved to [`best_model.pth`](best_model.pth).

---

## 📊 Evaluate

```sh
python -m src.evaluate
```

---

## 📤 Export to ONNX

```sh
python -m src.onnx_export
```

Output: [`plant_disease_classifier.onnx`](plant_disease_classifier.onnx)

---

## 🔍 Run Inference (CLI)

```sh
python -m src.inference.predict --help
```

Or use the convenience runner:

```sh
python run.py
```

---

## 🌐 Run the API

```sh
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the auto-generated Swagger UI.

---

## 🧪 Testing

```sh
python test.py
python sampletest.py
```

Or open [`src/test.ipynb`](src/test.ipynb) in Jupyter for interactive experimentation.

---

## 📦 Pre-trained Artifacts

| File | Description |
|------|-------------|
| [`best_model.pth`](best_model.pth) | PyTorch model weights |
| [`plant_disease_classifier.onnx`](plant_disease_classifier.onnx) | ONNX exported model |

---