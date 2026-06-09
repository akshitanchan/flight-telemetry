from fastapi.testclient import TestClient
from ml.serve import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "fake_mode": True}

def test_predict_fake_mode():
    payload = {
        "flight_id": "prc_test_999",
        "duration_s": 900.0,  # 15 mins
        "alt_change": 1000.0,
        "avg_speed": 250.0,
        "max_vrate": 10.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["flight_id"] == "prc_test_999"
    assert data["model_version"] == "fake-heuristic-v1"
    assert "predicted_fuel_kg" in data
    assert "latency_ms" in data
    
    # Fake heuristic: 15 mins * 50kg/min = 750kg
    assert abs(data["predicted_fuel_kg"] - 750.0) < 0.1
