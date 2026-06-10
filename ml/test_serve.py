"""
ml/test_serve.py — Offline test suite for the fuel-burn serving endpoint.
=========================================================================

All tests are designed to run with NO database and NO MLflow server.  The
real-model path uses a freshly-initialised FuelBurnMLP (randomly-weighted)
saved to a temp file, exercising the full torch.load → forward-pass path.

Test categories
---------------
test_health_check            — /health returns 200 + expected schema.
test_predict_fake_mode       — fake heuristic correctness (pre-existing).
test_predict_full_fields     — wider request schema; optional fields present.
test_predict_real_model      — loads a real FuelBurnMLP via ML_MODEL_PATH;
                               asserts numeric output and model_version.
test_predict_unknown_aircraft— unknown aircraft type routes to __unknown__ bucket.
test_logging_no_db           — _log_prediction silently skips when DB is absent.
test_logging_with_db         — _log_prediction inserts a row when DB is reachable
                               (skipped automatically if DATABASE_URL is unset).
test_p99_latency_slo         — SLO: p99 < 50 ms for single CPU inference on the
                               real FuelBurnMLP (N=200 requests, deterministic).
"""

import os
import time
import tempfile
import importlib
import numpy as np
import pytest
import torch

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_model() -> str:
    """Create a freshly-initialised FuelBurnMLP and save it to a temp .pth file.

    Returns the absolute path to the saved file.  The caller is responsible for
    cleanup (use tmp_path fixture or manual os.unlink).
    """
    from ml.model import FuelBurnMLP
    from ml.features import INPUT_DIM

    torch.manual_seed(42)
    model = FuelBurnMLP(input_dim=INPUT_DIM)
    model.eval()

    fd, path = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    torch.save(model, path)
    return path


def _minimal_payload(flight_id: str = "test_001") -> dict:
    """Return the smallest valid PredictRequest payload (required fields only)."""
    return {
        "flight_id": flight_id,
        "duration_s": 900.0,
        "alt_change": 1000.0,
        "avg_speed": 250.0,
        "max_vrate": 10.0,
    }


def _full_payload(flight_id: str = "test_002") -> dict:
    """Return a fully-populated PredictRequest payload including all optional fields."""
    return {
        "flight_id": flight_id,
        "icao24": "4b1800",
        "event_ts": "2025-06-01T12:00:00Z",
        "duration_s": 1800.0,
        "alt_change": -500.0,
        "avg_speed": 430.0,
        "max_vrate": 22.0,
        "avg_altitude": 35000.0,
        "max_altitude": 37000.0,
        "alt_std": 120.0,
        "avg_track_change": 0.5,
        "avg_mach": 0.78,
        "avg_tas": 450.0,
        "avg_cas": 280.0,
        "aircraft_type": "B738",
    }


# ---------------------------------------------------------------------------
# Env patching helpers
# ---------------------------------------------------------------------------

def _patch_env(overrides: dict) -> dict:
    """Apply env overrides; return dict of previous values for restoration."""
    old: dict = {}
    for k, v in overrides.items():
        old[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return old


def _restore_env(old: dict):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def temp_model_path():
    """Session-scoped real FuelBurnMLP checkpoint (38-feature, random weights)."""
    path = _make_temp_model()
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# Tests — fake mode (smoke / pre-existing)
# ---------------------------------------------------------------------------

def test_health_check():
    """Health endpoint returns 200 with expected schema in fake mode."""
    from ml.serve import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "fake_mode" in data


def test_predict_fake_mode():
    """Fake heuristic: 15-min interval → 750 kg (50 kg/min)."""
    from ml.serve import app
    client = TestClient(app)
    payload = _minimal_payload("prc_test_999")
    payload["duration_s"] = 900.0  # 15 minutes

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["flight_id"] == "prc_test_999"
    assert data["model_version"] == "fake-heuristic-v1"
    assert "predicted_fuel_kg" in data
    assert "latency_ms" in data
    assert abs(data["predicted_fuel_kg"] - 750.0) < 0.1


def test_predict_full_fields():
    """Full payload (all optional fields) is accepted in fake mode with 200."""
    from ml.serve import app
    client = TestClient(app)
    response = client.post("/predict", json=_full_payload())
    assert response.status_code == 200
    data = response.json()
    assert "predicted_fuel_kg" in data
    assert data["model_version"] == "fake-heuristic-v1"


def test_predict_unknown_aircraft():
    """Unknown aircraft type in fake mode does not raise; heuristic is deterministic."""
    from ml.serve import app
    client = TestClient(app)
    payload = _minimal_payload("test_unk")
    payload["aircraft_type"] = "ZZUNKNOWN"  # not in AIRCRAFT_TYPES
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Tests — real model (ML_MODEL_PATH)
#
# TestClient must be used as a context manager so that the FastAPI lifespan
# (which populates MODULE-level MODEL) is properly entered/exited.
# We reload ml.serve after patching env to get a fresh module state.
# ---------------------------------------------------------------------------

def test_predict_real_model(temp_model_path):
    """Real FuelBurnMLP loaded via ML_MODEL_PATH returns a finite float output."""
    old_env = _patch_env({
        "ML_FAKE_MODE": "false",
        "ML_MODEL_PATH": temp_model_path,
        "ML_MODEL_URI": None,
    })
    try:
        import ml.serve as serve_mod
        importlib.reload(serve_mod)

        with TestClient(serve_mod.app) as client:
            assert not serve_mod.FAKE_MODE, "Expected real-model mode after reload"
            assert serve_mod.MODEL is not None, "Model must be loaded by lifespan"

            payload = _full_payload("real_001")
            response = client.post("/predict", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert data["flight_id"] == "real_001"
            assert data["model_version"] == temp_model_path
            fuel = data["predicted_fuel_kg"]
            assert isinstance(fuel, float)
            assert not (fuel != fuel)   # not NaN
            assert abs(fuel) < 1e6     # sanity: not exploding
    finally:
        _restore_env(old_env)


def test_predict_real_model_unknown_aircraft(temp_model_path):
    """Unknown aircraft_type routes to __unknown__ one-hot bucket; no crash."""
    old_env = _patch_env({
        "ML_FAKE_MODE": "false",
        "ML_MODEL_PATH": temp_model_path,
        "ML_MODEL_URI": None,
    })
    try:
        import ml.serve as serve_mod
        importlib.reload(serve_mod)

        with TestClient(serve_mod.app) as client:
            payload = _full_payload("real_unk")
            payload["aircraft_type"] = "ZZUNKNOWN"
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            fuel = data["predicted_fuel_kg"]
            assert isinstance(fuel, float) and not (fuel != fuel)
    finally:
        _restore_env(old_env)


# ---------------------------------------------------------------------------
# Test — C2 logging (availability-gated)
# ---------------------------------------------------------------------------

def test_logging_no_db():
    """_log_prediction silently returns (no raise) when DATABASE_URL is absent."""
    import ml.serve as serve_mod

    # Temporarily unset DATABASE_URL to ensure healthcheck() returns False.
    orig = os.environ.pop("DATABASE_URL", None)
    try:
        # Must not raise under any circumstances.
        serve_mod._log_prediction(
            run_id="test-run",
            model_uri="local",
            icao24="abc123",
            event_ts="2025-06-01T00:00:00Z",
            predicted_fuel_kg=42.0,
        )
    finally:
        if orig is not None:
            os.environ["DATABASE_URL"] = orig


@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping live DB test",
)
def test_logging_with_db():
    """When a DB is reachable, _log_prediction inserts a row in ml_predictions."""
    import ml.serve as serve_mod
    from shared.store.pg import get_conn, healthcheck

    if not healthcheck():
        pytest.skip("DB healthcheck failed — skipping live DB test")

    # Record row count before the test write.
    with get_conn() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM ml_predictions WHERE run_id = %s",
            ("pytest-logging-test",),
        ).fetchone()[0]

    serve_mod._log_prediction(
        run_id="pytest-logging-test",
        model_uri="test-uri",
        icao24="4btest",
        event_ts="2025-06-01T12:00:00Z",
        predicted_fuel_kg=123.45,
    )

    with get_conn() as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM ml_predictions WHERE run_id = %s",
            ("pytest-logging-test",),
        ).fetchone()[0]

    assert after == before + 1, f"Expected one new row, got {after - before}"


# ---------------------------------------------------------------------------
# Test — p99 latency SLO
# ---------------------------------------------------------------------------

def test_p99_latency_slo(temp_model_path):
    """SLO: p99 inference latency < 50 ms on CPU (N=200 single requests).

    Methodology:
    - Load a real FuelBurnMLP (38-feature, random weights, CPU).
    - Send N_REQUESTS sequential single-item requests via TestClient (no network).
    - Measure wall-clock time around each call using time.perf_counter().
    - Compute p99 and assert < SLO_MS.

    The TestClient bypasses ASGI networking so measured time is dominated by
    PyTorch forward-pass + Pydantic serialisation, which is the relevant CPU
    inference budget.  No DB or MLflow server needed.
    """
    N_REQUESTS = 200
    SLO_MS = 50.0  # p99 must be below this value

    old_env = _patch_env({
        "ML_FAKE_MODE": "false",
        "ML_MODEL_PATH": temp_model_path,
        "ML_MODEL_URI": None,
    })
    try:
        import ml.serve as serve_mod
        importlib.reload(serve_mod)

        with TestClient(serve_mod.app) as client:
            assert serve_mod.MODEL is not None, "Model must be loaded for SLO test"

            payload = _full_payload("slo_test")

            # Warm-up: a few requests before the timed window to avoid cold-start
            # JIT compilation penalties on the first forward pass.
            for _ in range(5):
                client.post("/predict", json=payload)

            latencies_ms = []
            for i in range(N_REQUESTS):
                payload["flight_id"] = f"slo_{i}"
                t0 = time.perf_counter()
                resp = client.post("/predict", json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                assert resp.status_code == 200
                latencies_ms.append(elapsed_ms)

        p99 = float(np.percentile(latencies_ms, 99))
        p50 = float(np.percentile(latencies_ms, 50))
        p_max = float(np.max(latencies_ms))
        print(
            f"\nLatency over {N_REQUESTS} requests: "
            f"p50={p50:.2f} ms  p99={p99:.2f} ms  max={p_max:.2f} ms"
        )
        assert p99 < SLO_MS, (
            f"p99 latency {p99:.2f} ms exceeds SLO of {SLO_MS} ms "
            f"(p50={p50:.2f} ms, max={p_max:.2f} ms)"
        )
    finally:
        _restore_env(old_env)
