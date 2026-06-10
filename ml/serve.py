import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import torch

from shared.obs.telemetry import (
    ml_predict_latency_seconds,
    http_requests_total,
    setup_metrics,
    setup_tracing,
    get_tracer,
)

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO  [%(name)s] %(message)s")
logger = logging.getLogger("ml.serve")

# Configure tracing early — before the app is used.  No-ops gracefully when no
# collector is reachable (offline runs and CI stay green).
setup_tracing("ml-serve")
_tracer = get_tracer("ml.serve")


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


# --- Config (env-overridable) ---
# FAKE_MODE serves a fast heuristic when no trained model is mounted. Set
# ML_FAKE_MODE=false and ML_MODEL_URI=<mlflow uri> to serve the real model.
FAKE_MODE = _env_bool("ML_FAKE_MODE", True)
MODEL_URI = os.environ.get("ML_MODEL_URI")  # e.g. "models:/FuelBurn/Production" or "runs:/<id>/model"
MODEL = None


def _load_model():
    """Load the PyTorch model from MLflow; return it, or None (and flip to fake) on failure."""
    global FAKE_MODE
    if FAKE_MODE:
        return None
    if not MODEL_URI:
        logger.warning("ML_MODEL_URI not set; falling back to fake mode")
        FAKE_MODE = True
        return None
    try:
        import mlflow.pytorch
        model = mlflow.pytorch.load_model(MODEL_URI)
        model.eval()
        logger.info("Loaded model from %s", MODEL_URI)
        return model
    except Exception as e:  # noqa: BLE001 — any load failure should degrade gracefully
        logger.warning("Model load failed (%s); falling back to fake mode", e)
        FAKE_MODE = True
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup (replaces the deprecated on_event hook)."""
    global MODEL
    MODEL = _load_model()
    logger.info("ML serving ready. Fake mode: %s", FAKE_MODE)
    yield


app = FastAPI(title="Fuel Burn Estimation API", version="0.1.0", lifespan=lifespan)

# Mount /metrics and register the HTTP metrics + latency logging middleware.
# setup_metrics is idempotent so test re-imports are safe.
setup_metrics(app)


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


@app.middleware("http")
async def log_latency_middleware(request: Request, call_next):
    """Log per-request latency to structured logger.

    Prometheus HTTP counters are handled separately by the metrics middleware
    registered in setup_metrics; this middleware focuses only on structured
    logging so the two concerns remain independent.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - Latency: {process_time_ms:.2f} ms")
    return response


@app.get("/health")
def health_check():
    return {"status": "ok", "fake_mode": FAKE_MODE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    with _tracer.start_as_current_span("ml.predict") as span:
        span.set_attribute("flight_id", req.flight_id)

        start_time = time.perf_counter()

        # Guard: serve the heuristic whenever we're in fake mode OR no model is loaded.
        if FAKE_MODE or MODEL is None:
            # Simple heuristic for fake mode: ~50kg per minute of duration
            pred_fuel = (req.duration_s / 60.0) * 50.0
            version = "fake-heuristic-v1"
        else:
            # Real model inference. Shape [1, 4] (batch of one) so the output is
            # well-defined rather than relying on PyTorch auto-broadcast (audit M7).
            features = torch.tensor([[
                req.duration_s,
                req.alt_change,
                req.avg_speed,
                req.max_vrate,
            ]], dtype=torch.float32)

            with torch.no_grad():
                pred = MODEL(features)

            pred_fuel = float(pred.reshape(-1)[0].item())
            version = "mlflow-baseline"

        elapsed_s = time.perf_counter() - start_time
        ml_predict_latency_seconds.observe(elapsed_s)
        process_time_ms = elapsed_s * 1000

        span.set_attribute("model_version", version)
        span.set_attribute("predicted_fuel_kg", pred_fuel)

        return PredictResponse(
            flight_id=req.flight_id,
            predicted_fuel_kg=pred_fuel,
            model_version=version,
            latency_ms=process_time_ms,
        )
