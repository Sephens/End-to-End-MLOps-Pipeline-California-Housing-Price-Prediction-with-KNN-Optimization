"""
Integration tests for app/main.py (FastAPI endpoints).

These tests use a mock model so no trained .pkl file is required.
The mock is injected directly into the app module — no pickling needed.
"""

import pickle
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """A lightweight mock that returns a fixed prediction."""
    m = MagicMock()
    m.predict.return_value = np.array([2.5])
    return m


@pytest.fixture
def client(mock_model, tmp_path):
    """
    TestClient with the model injected directly into the app module globals.
    Bypasses the lifespan loader entirely so no .pkl file is needed.
    """
    import app.main as main_module

    # Patch os.path.exists so the lifespan startup check passes,
    # and patch open+pickle.load so it returns our mock instead of reading disk.
    with patch("app.main.os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=b"")), \
         patch("pickle.load", return_value=mock_model), \
         patch.object(main_module, "model", mock_model):
        with TestClient(main_module.app) as c:
            yield c


@pytest.fixture
def valid_payload():
    return {
        "MedInc":     3.5,
        "HouseAge":   25.0,
        "AveRooms":   5.2,
        "AveBedrms":  1.1,
        "Population": 800.0,
        "AveOccup":   3.0,
        "Latitude":   37.5,
        "Longitude":  -120.5,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_status_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_message_in_response(self, client):
        response = client.get("/")
        assert "message" in response.json()


class TestHealthEndpoint:
    def test_status_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


class TestPredictEndpoint:
    def test_status_200(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_response_contains_prediction(self, client, valid_payload):
        data = client.post("/predict", json=valid_payload).json()
        assert "predicted_value_100k" in data
        assert "predicted_value_usd" in data

    def test_prediction_is_numeric(self, client, valid_payload):
        data = client.post("/predict", json=valid_payload).json()
        assert isinstance(data["predicted_value_100k"], float)

    def test_usd_format(self, client, valid_payload):
        data = client.post("/predict", json=valid_payload).json()
        assert data["predicted_value_usd"].startswith("$")

    def test_missing_field_returns_422(self, client, valid_payload):
        del valid_payload["MedInc"]
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422

    def test_invalid_latitude_returns_422(self, client, valid_payload):
        valid_payload["Latitude"] = 99.0   # out of range
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422

    def test_negative_population_returns_422(self, client, valid_payload):
        valid_payload["Population"] = -1.0
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422
