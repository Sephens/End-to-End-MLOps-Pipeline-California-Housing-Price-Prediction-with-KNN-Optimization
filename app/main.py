"""
FastAPI application serving the California Housing price prediction model.

Endpoints:
    GET  /          — Health check
    GET  /health    — Detailed health status
    POST /predict   — Predict median house value

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import pickle
import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "california_knn_pipeline.pkl")

# Global model handle (loaded at startup)
model = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; train it first if the .pkl is missing."""
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model file not found at '{MODEL_PATH}'. Training now...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.train"],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"Training failed:\n{result.stderr}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from '{MODEL_PATH}'")
    yield
    model = None
    print("🛑 Model unloaded.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="California Housing Price Predictor",
    description=(
        "Predicts the **median house value** (in $100k units) for a California "
        "district based on census features. Built with Scikit-Learn KNN + GridSearchCV."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class HouseFeatures(BaseModel):
    """Input features matching the California Housing dataset."""

    MedInc:     float = Field(..., gt=0,           description="Median income ($10k)")
    HouseAge:   float = Field(..., ge=0,            description="Median house age (years)")
    AveRooms:   float = Field(..., gt=0,            description="Avg rooms per household")
    AveBedrms:  float = Field(..., gt=0,            description="Avg bedrooms per household")
    Population: float = Field(..., gt=0,            description="Block group population")
    AveOccup:   float = Field(..., gt=0,            description="Avg household occupancy")
    Latitude:   float = Field(..., ge=32,  le=42,   description="Latitude")
    Longitude:  float = Field(..., ge=-125, le=-114, description="Longitude")

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 3.5, "HouseAge": 25.0, "AveRooms": 5.2,
                "AveBedrms": 1.1, "Population": 800.0, "AveOccup": 3.0,
                "Latitude": 37.5, "Longitude": -120.5,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_value_100k: float = Field(..., description="Predicted value in $100k")
    predicted_value_usd:  str   = Field(..., description="Human-readable USD value")
    model_version:        str   = "1.0.0"


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_path:   str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"message": "California Housing Price Predictor API is running 🏠"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: HouseFeatures):
    """
    Predict the median house value for the given district features.
    Returns the value in $100k units (e.g. 2.41 = $241,000).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet — try again in a moment.")

    input_df = pd.DataFrame([features.model_dump()])

    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        predicted_value_100k=round(prediction, 4),
        predicted_value_usd=f"${prediction * 100_000:,.0f}",
    )