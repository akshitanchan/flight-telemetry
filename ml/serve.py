"""
ml/serve.py — Fuel-Burn Estimation serving endpoint.
======================================================

Environment variables
---------------------
ML_FAKE_MODE        (default "true")  — Set to "false" to load a real model.
ML_MODEL_PATH       (no default)      — Path to a local .pth checkpoint saved
                                        with torch.save(model, path).  Loaded
                                        first when ML_FAKE_MODE=false.
ML_MODEL_URI        (no default)      — MLflow model URI, e.g.
                                        "models:/FuelBurn@production" (ml-09
                                        alias-based registry; use the alias
                                        form, not the deprecated stage form).
                                        Used as fallback when ML_MODEL_PATH is
                                        absent.  Both vars are ignored in fake
                                        mode.
ML_RUN_ID           (default "serve") — run_id written to ml_predictions rows.

Model loading priority (ML_FAKE_MODE=false):
  1. ML_MODEL_PATH  → torch.load (no MLflow server needed; use for offline tests)
  2. ML_MODEL_URI   → mlflow.pytorch.load_model (needs a running tracking server)
  3. Neither set     → warning + fall back to fake mode

Request schema (PredictRequest)
--------------------------------
The 38-dimensional input vector is assembled server-side in FEATURE_COLUMNS order:
  [ duration_s, alt_change, avg_speed, max_vrate, avg_altitude, max_altitude,
    alt_std, avg_track_change, avg_mach, avg_tas, avg_cas,   ← 11 numeric
    ac_A20N, ac_A21N, ..., ac___unknown__ ]                  ← 27 one-hot

Clients send the 11 numeric fields by name + an `aircraft_type` string.
The server calls encode_aircraft_type(aircraft_type) to produce the 27-dim
one-hot vector and concatenates it to obtain the full [38] input.

C2 prediction logging
---------------------
One row is written to ml_predictions after each successful inference.  The
write is availability-gated: if shared.store.pg.healthcheck() returns False
(no DATABASE_URL, DB unreachable, etc.) the write is silently skipped — the
prediction response is never delayed or blocked by DB issues.
"""

import os
import time
import logging
import datetime
from contextlib import asynccontextmanager
from typing import Optional

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
from ml.features import (
    FEATURE_COLUMNS,
    INPUT_DIM,
    encode_aircraft_type,
)
from ml.model import FuelBurnMLP

# ---------------------------------------------------------------------------
# Logging + tracing
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO  [%(name)s] %(message)s")
logger = logging.getLogger("ml.serve")

# No-ops gracefully when no collector is reachable (offline / CI).
setup_tracing("ml-serve")
_tracer = get_tracer("ml.serve")


# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


# FAKE_MODE: serves a fast heuristic with no model weights.
# Set ML_FAKE_MODE=false + one of ML_MODEL_PATH / ML_MODEL_URI to serve real.
FAKE_MODE: bool = _env_bool("ML_FAKE_MODE", True)

# ML_MODEL_PATH: local .pth file — preferred for offline / test environments.
# Loaded with torch.load(MODEL_PATH) — no MLflow server required.
MODEL_PATH: Optional[str] = os.environ.get("ML_MODEL_PATH")

# ML_MODEL_URI: MLflow model URI e.g. "models:/FuelBurn@production".
# ml-09 (registry promotion) registers the alias-based URI.
# Wire it here so that once ml-09 ships, operators just set this env var.
MODEL_URI: Optional[str] = os.environ.get("ML_MODEL_URI")

# ML_RUN_ID: written to ml_predictions.run_id so predictions can be joined
# back to a training run.  Override with the MLflow run_id at deploy time.
RUN_ID: str = os.environ.get("ML_RUN_ID", "serve")

# Module-level handle; populated during lifespan startup.
MODEL: Optional[FuelBurnMLP] = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model() -> Optional[FuelBurnMLP]:
    """Load the model; return it, or None (and flip FAKE_MODE) on failure.

    Priority:
      1. ML_MODEL_PATH  → torch.load (offline-safe, no tracking server)
      2. ML_MODEL_URI   → mlflow.pytorch.load_model
      3. Neither set    → warning + fake mode
    """
    global FAKE_MODE
    if FAKE_MODE:
        return None

    if MODEL_PATH:
        try:
            model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            if not isinstance(model, FuelBurnMLP):
                # Handle state-dict checkpoints produced by some training runs.
                m = FuelBurnMLP(input_dim=INPUT_DIM)
                m.load_state_dict(model)
                model = m
            model.eval()
            logger.info("Loaded model from local path %s", MODEL_PATH)
            return model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local model load failed (%s); trying MODEL_URI", exc)

    if MODEL_URI:
        try:
            import mlflow.pytorch  # type: ignore[import]
            model = mlflow.pytorch.load_model(MODEL_URI)
            model.eval()
            logger.info("Loaded model from MLflow URI %s", MODEL_URI)
            return model
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow model load failed (%s); falling back to fake mode", exc)
            FAKE_MODE = True
            return None

    logger.warning(
        "Neither ML_MODEL_PATH nor ML_MODEL_URI is set; falling back to fake mode"
    )
    FAKE_MODE = True
    return None


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    global MODEL
    MODEL = _load_model()
    logger.info("ML serving ready. fake_mode=%s, input_dim=%d", FAKE_MODE, INPUT_DIM)
    yield


app = FastAPI(title="Fuel Burn Estimation API", version="0.2.0", lifespan=lifespan)

# Mount /metrics and register HTTP metrics middleware.  Idempotent.
setup_metrics(app)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

# Numeric feature field names, in the same order as FEATURE_COLUMNS[:11].
# Keeping them explicit here lets Pydantic generate accurate OpenAPI docs.
_NUMERIC_FIELDS = FEATURE_COLUMNS[:11]  # the 11 trajectory-derived features


class PredictRequest(BaseModel):
    """Inference request.

    The client sends the 11 trajectory-derived numeric features by name and an
    aircraft_type string.  The server builds the 38-dim input vector server-side:
      [ duration_s, alt_change, avg_speed, max_vrate, avg_altitude, max_altitude,
        alt_std, avg_track_change, avg_mach, avg_tas, avg_cas,   ← indices 0-10
        ac_A20N, ac_A21N, ..., ac___unknown__ ]                  ← indices 11-37
    as defined in ml.features.FEATURE_COLUMNS.

    Optional fields for C2 logging:
      icao24     — transponder hex code (written to ml_predictions.icao24)
      event_ts   — UTC ISO-8601 timestamp of the flight event
    """

    # ----- Identity (optional for logging) -----
    flight_id: str = Field(..., description="Unique flight identifier")
    icao24: Optional[str] = Field(None, description="Transponder hex code (for C2 logging)")
    event_ts: Optional[str] = Field(
        None, description="UTC ISO-8601 event timestamp (for C2 logging)"
    )

    # ----- Trajectory-derived numeric features (11) -----
    duration_s: float = Field(..., description="Interval duration in seconds")
    alt_change: float = Field(..., description="Altitude change over interval (ft)")
    avg_speed: float = Field(..., description="Mean groundspeed over interval (kts)")
    max_vrate: float = Field(..., description="Max |vertical_rate| over interval (ft/min)")
    avg_altitude: float = Field(0.0, description="Mean barometric altitude (ft); fill 0.0")
    max_altitude: float = Field(0.0, description="Max altitude over interval (ft); fill 0.0")
    alt_std: float = Field(0.0, description="Std-dev of altitude (ft); fill 0.0 if < 2 pts")
    avg_track_change: float = Field(
        0.0, description="Mean absolute per-step track change (deg/step); fill 0.0"
    )
    avg_mach: float = Field(0.0, description="Mean Mach number; fill 0.0 when all NaN")
    avg_tas: float = Field(0.0, description="Mean True Air Speed (kts); fill 0.0")
    avg_cas: float = Field(0.0, description="Mean Calibrated Air Speed (kts); fill 0.0")

    # ----- Aircraft type (encoded server-side → 27 one-hot dims) -----
    aircraft_type: str = Field(
        "__unknown__",
        description=(
            "ICAO aircraft type designator (e.g. 'B738', 'A320').  "
            "Unknown types map to the __unknown__ bucket.  "
            "Encoded server-side via ml.features.encode_aircraft_type()."
        ),
    )


class PredictResponse(BaseModel):
    flight_id: str
    predicted_fuel_kg: float
    model_version: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Feature assembly helper
# ---------------------------------------------------------------------------

def _build_feature_tensor(req: PredictRequest) -> torch.Tensor:
    """Assemble the 38-dim input tensor in FEATURE_COLUMNS order.

    The 11 numeric values are taken in the order defined by FEATURE_COLUMNS
    (duration_s, alt_change, avg_speed, ..., avg_cas), then the 27-dim one-hot
    from encode_aircraft_type() is appended — matching the training pipeline.
    """
    numeric = [
        req.duration_s,
        req.alt_change,
        req.avg_speed,
        req.max_vrate,
        req.avg_altitude,
        req.max_altitude,
        req.alt_std,
        req.avg_track_change,
        req.avg_mach,
        req.avg_tas,
        req.avg_cas,
    ]
    one_hot = encode_aircraft_type(req.aircraft_type)
    features = numeric + one_hot
    assert len(features) == INPUT_DIM, (
        f"Feature vector length {len(features)} != INPUT_DIM {INPUT_DIM}"
    )
    return torch.tensor([features], dtype=torch.float32)  # shape [1, 38]


# ---------------------------------------------------------------------------
# C2 prediction logging (availability-gated)
# ---------------------------------------------------------------------------

def _log_prediction(
    run_id: str,
    model_uri: str,
    icao24: str,
    event_ts: Optional[str],
    predicted_fuel_kg: float,
) -> None:
    """Write one row to ml_predictions.  Silently skips if DB is unreachable.

    Contract: this function MUST NOT raise — it must never delay or crash the
    /predict response.  All exceptions are caught and logged at WARNING level.
    """
    try:
        from shared.store.pg import get_conn, healthcheck

        if not healthcheck():
            logger.debug("DB not reachable; skipping ml_predictions write")
            return

        ts: Optional[datetime.datetime] = None
        if event_ts:
            try:
                ts = datetime.datetime.fromisoformat(event_ts.replace("Z", "+00:00"))
            except ValueError:
                logger.warning("Invalid event_ts %r; writing NULL", event_ts)

        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO ml_predictions
                    (run_id, model_uri, icao24, event_ts, predicted_fuel_kg, actual_fuel_kg)
                VALUES (%s, %s, %s, %s, %s, NULL)
                """,
                (run_id, model_uri, icao24 or "", ts, predicted_fuel_kg),
            )
            conn.commit()
        logger.debug("Logged prediction to ml_predictions (icao24=%s)", icao24)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ml_predictions write skipped: %s", exc)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

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
    logger.info("%s %s - Latency: %.2f ms", request.method, request.url.path, process_time_ms)
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "fake_mode": FAKE_MODE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    with _tracer.start_as_current_span("ml.predict") as span:
        span.set_attribute("flight_id", req.flight_id)

        start_time = time.perf_counter()

        if FAKE_MODE or MODEL is None:
            # Heuristic fallback: ~50 kg per minute of interval duration.
            pred_fuel = (req.duration_s / 60.0) * 50.0
            version = "fake-heuristic-v1"
        else:
            # Real model inference.
            # _build_feature_tensor produces shape [1, INPUT_DIM] (batched) so
            # MODEL(features) returns a scalar tensor; squeeze(-1) is safe for
            # both batched and unbatched inputs (see FuelBurnMLP.forward).
            features = _build_feature_tensor(req)
            with torch.no_grad():
                pred = MODEL(features)
            pred_fuel = float(pred.reshape(-1)[0].item())
            version = MODEL_URI or MODEL_PATH or "local-checkpoint"

        elapsed_s = time.perf_counter() - start_time
        ml_predict_latency_seconds.observe(elapsed_s)
        process_time_ms = elapsed_s * 1000

        span.set_attribute("model_version", version)
        span.set_attribute("predicted_fuel_kg", pred_fuel)

        # C2 availability-gated logging — fire-and-forget, never blocks response.
        _log_prediction(
            run_id=RUN_ID,
            model_uri=MODEL_URI or MODEL_PATH or "local",
            icao24=req.icao24,
            event_ts=req.event_ts,
            predicted_fuel_kg=pred_fuel,
        )

        return PredictResponse(
            flight_id=req.flight_id,
            predicted_fuel_kg=pred_fuel,
            model_version=version,
            latency_ms=process_time_ms,
        )
