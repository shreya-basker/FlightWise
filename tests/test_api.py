from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint_is_available():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "model_ready" in response.json()


def test_live_configuration_does_not_expose_a_secret():
    client = TestClient(app)
    response = client.get("/api/live/config")
    assert response.status_code == 200
    assert set(response.json()) == {"flight_lookup_configured", "weather_provider"}
