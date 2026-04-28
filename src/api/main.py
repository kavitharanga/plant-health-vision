from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import tempfile, shutil, os, sys

# Make sure inference/ is importable from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import PlantDiseaseClassifier
from api.schemas import Prediction, PredictResponse

# ── App & model init ──────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Classifier",
    description="Syngenta - ONNX-based plant disease detection API",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "plant_disease_classifier.onnx")
DEVICE = os.getenv("DEVICE", "cpu")

clf = PlantDiseaseClassifier(MODEL_PATH, device=DEVICE)


# ── Routes ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=3, ge=1, le=10),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = clf.predict(tmp_path, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return PredictResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        is_healthy=result["is_healthy"],
        top_k=[
            Prediction(class_name=p["class"], confidence=p["confidence"])
            for p in result["top_k"]
        ],
        latency_ms=result["latency_ms"],
    )
