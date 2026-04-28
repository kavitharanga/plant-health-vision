from pydantic import BaseModel
from typing import List


class Prediction(BaseModel):
    class_name: str
    confidence: float


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_healthy: bool
    top_k: List[Prediction]
    latency_ms: float
