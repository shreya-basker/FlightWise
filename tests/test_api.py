import httpx
import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.main import app
from backend.ml.train_us import train
from tests.test_us_pipeline import synthetic_training_frame

client = TestClient(app)
WEATHER = {"airport": "ATL", "observed_at": "2024-01-10T09:00", "timezone": "America/New_York", "kind": "forecast", "source": "test double",
           "TemperatureC": 4.0, "PrecipitationMM": 0.0, "WindSpeedKPH": 12.0, "CloudCoverPct": 80, "VisibilityM": 20000}
PAYLOAD = {"airline": "AA", "origin": "ATL", "destination": "LAX", "flight_date": "2024-01-10", "scheduled_departure_hour": 9}


@pytest.fixture
def us_model(tmp_path, monkeypatch):
    data = tmp_path / "clean.csv"
    synthetic_training_frame().to_csv(data, index=False)
    train(data, tmp_path / "model.joblib", tmp_path / "metrics.json")
    monkeypatch.setattr(main, "US_MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(main, "US_METRICS_PATH", tmp_path / "metrics.json")
    main.load_us_artifacts.cache_clear()
    yield
    main.load_us_artifacts.cache_clear()


@pytest.fixture
def no_us_model(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "US_MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(main, "US_METRICS_PATH", tmp_path / "missing.json")
    main.load_us_artifacts.cache_clear()
    yield
    main.load_us_artifacts.cache_clear()


def fail_if_called(*args, **kwargs):
    raise AssertionError("an external provider must not be called")


def test_health_endpoint_is_available():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert {"status", "us_model_ready", "model_ready"} <= set(response.json())


def test_frontend_is_served():
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_live_configuration_does_not_expose_a_secret(monkeypatch):
    monkeypatch.setenv("AVIATIONSTACK_API_KEY", "secret-test-value")
    response = client.get("/api/live/config")
    assert response.status_code == 200
    assert set(response.json()) == {"flight_lookup_configured", "weather_provider"}
    assert "secret-test-value" not in response.text


def test_missing_us_model_returns_503_before_weather_call(no_us_model, monkeypatch):
    monkeypatch.setattr(main, "departure_weather", fail_if_called)
    response = client.post("/api/us/predict", json=PAYLOAD)
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]
    assert client.get("/api/us/model-info").status_code == 503


def test_predict_returns_estimate_with_weather_and_disclaimer(us_model, monkeypatch):
    monkeypatch.setattr(main, "departure_weather", lambda airport, day, hour: {**WEATHER, "airport": airport})
    response = client.post("/api/us/predict", json=PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction"] in {"likely_delayed", "likely_on_time"}
    assert 0 <= body["delay_probability"] <= 1
    assert body["distance_km"] > 3000 and body["distance_source"].startswith("great-circle")
    assert body["weather_context"]["source"] == "test double"
    assert "not a flight status" in body["disclaimer"]
    assert body["warnings"] == []  # AA, ATL, LAX and January are all in the synthetic training data


def test_invalid_airport_returns_422_without_weather_call(us_model, monkeypatch):
    monkeypatch.setattr(main, "departure_weather", fail_if_called)
    response = client.post("/api/us/predict", json={**PAYLOAD, "origin": "ZZZ"})
    assert response.status_code == 422
    assert "ZZZ" in response.json()["detail"]
    assert client.post("/api/us/predict", json={**PAYLOAD, "destination": "ATL"}).status_code == 422
    assert client.post("/api/us/predict", json={**PAYLOAD, "scheduled_departure_hour": 25}).status_code == 422


def test_weather_provider_failure_returns_502(us_model, monkeypatch):
    def down(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(main, "departure_weather", down)
    response = client.post("/api/us/predict", json=PAYLOAD)
    assert response.status_code == 502
    assert "Open-Meteo" in response.json()["detail"]


def test_live_lookup_without_key_returns_503(us_model, monkeypatch, tmp_path):
    monkeypatch.delenv("AVIATIONSTACK_API_KEY", raising=False)
    monkeypatch.setenv("FLIGHTWISE_ENV_FILE", str(tmp_path / "absent.env"))
    response = client.get("/api/live/flight/AA100")
    assert response.status_code == 503
    assert "API key required" in response.json()["detail"]


def test_live_lookup_rejects_malformed_flight_number(us_model):
    assert client.get("/api/live/flight/not-a-flight").status_code == 422
