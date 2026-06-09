import time
import logging
from typing import Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import torch

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO  [%(name)s] %(message)s")
logger = logging.getLogger("ml.serve")

app = FastAPI(title="Fuel Burn Estimation API", version="0.1.0")

class PredictRequest(BaseModel):
    flight_id: str = Field(..., description="Unique flight identifier")
    duration_s: float = Field(..., description="Interval duration in seconds")
    alt_change: float = Field(..., description="Altitude change in meters")
    avg_speed: float = Field(..., description="Average ground speed")
    max_vrate: float = Field(..., description="Maximum vertical rate")

class PredictResponse(BaseModel):
    flight_id: str
    predicted_fuel_kg: float
    model_version: str
    latency_ms: float

# Global model state
MODEL = None
FAKE_MODE = True

@app.on_event("startup")
async def startup_event():
    global MODEL, FAKE_MODE
    # In a real environment, we'd load the MLflow model from a path provided via env var.
    # For the scaffold, we default to fake mode unless wired otherwise.
    logger.info(f"Starting ML Serving. Fake Mode: {FAKE_MODE}")
    if not FAKE_MODE:
        # Placeholder for real model load
        pass

@app.middleware("http")
async def log_latency_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Simple latency logging
    logger.info(f"{request.method} {request.url.path} - Latency: {process_time_ms:.2f} ms")
    return response

@app.get("/health")
def health_check():
    return {"status": "ok", "fake_mode": FAKE_MODE}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start_time = time.perf_counter()
    
    if FAKE_MODE:
        # Simple heuristic for fake mode: ~50kg per minute of duration
        pred_fuel = (req.duration_s / 60.0) * 50.0
        version = "fake-heuristic-v1"
    else:
        # Real model inference
        features = torch.tensor([
            req.duration_s, 
            req.alt_change, 
            req.avg_speed, 
            req.max_vrate
        ], dtype=torch.float32)
        
        with torch.no_grad():
            pred = MODEL(features)
        
        pred_fuel = float(pred.item())
        version = "mlflow-baseline"
        
    process_time_ms = (time.perf_counter() - start_time) * 1000
    
    return PredictResponse(
        flight_id=req.flight_id,
        predicted_fuel_kg=pred_fuel,
        model_version=version,
        latency_ms=process_time_ms
    )
