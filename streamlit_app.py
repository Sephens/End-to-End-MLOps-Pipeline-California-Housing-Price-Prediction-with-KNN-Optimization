"""
California Housing Price Predictor — self-contained Streamlit app.
Trains the model on first run, caches it, and serves predictions directly.
No separate API needed.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "california_knn_pipeline.pkl"
FEATURE_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
PARAM_GRID = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights":     ["uniform", "distance"],
    "knn__p":           [1, 2],
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CA Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)


# ── Model training & caching ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """Train the model and return (model, metrics). Always trains fresh."""

    # Build pipeline
    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
        ]), FEATURE_COLUMNS)],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsRegressor()),
    ])

    # Load data & split
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Grid search
    grid = GridSearchCV(pipeline, PARAM_GRID, cv=5, scoring="r2", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    metrics = {
        "r2":          round(float(r2_score(y_test, y_pred)), 4),
        "rmse":        round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "best_params": {k: v for k, v in grid.best_params_.items()},
        "cv_r2":       round(float(grid.best_score_), 4),
    }

    return best_model, metrics


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏠 CA Housing Predictor")
    st.markdown("---")

    st.subheader("About")
    st.markdown(
        """
        Predicts the **median house value** for a California district
        using a K-Nearest Neighbors model trained on the
        [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html).
        """
    )

    st.markdown("---")
    st.subheader("Model Info")

    with st.spinner("Training model — this takes ~2 mins on first load…"):
        model, metrics = load_or_train_model()

    st.success("Model ready!")
    st.metric("Test R²",   str(metrics["r2"]))
    st.metric("Test RMSE", f"{metrics['rmse']} ($100k)")
    st.metric("CV R²",     str(metrics["cv_r2"]))

    with st.expander("Best hyperparameters"):
        for k, v in metrics["best_params"].items():
            st.write(f"**{k.replace('knn__', '')}:** {v}")


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🏠 California Housing Price Predictor")
st.markdown(
    "Adjust the inputs below to describe a district, then click **Predict** "
    "to get the estimated median house value."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Economic & Housing")
    med_inc    = st.slider("Median Income ($10k)",       0.5,  15.0,  3.5,  step=0.1,
                           help="Median household income in the block group (in $10,000s)")
    house_age  = st.slider("House Age (years)",           1.0,  52.0, 25.0,  step=1.0)
    ave_rooms  = st.slider("Avg Rooms / Household",       1.0,  20.0,  5.2,  step=0.1)
    ave_bedrms = st.slider("Avg Bedrooms / Household",    0.5,   5.0,  1.1,  step=0.1)

with col2:
    st.subheader("👥 Population & Location")
    population = st.slider("Block Population",            3.0, 5000.0, 800.0, step=10.0)
    ave_occup  = st.slider("Avg Occupancy",               1.0,  10.0,  3.0,  step=0.1)
    latitude   = st.slider("Latitude",                   32.5,  42.0, 37.5,  step=0.1)
    longitude  = st.slider("Longitude",                -124.5,-114.3,-120.5,  step=0.1)

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────

if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "MedInc":     med_inc,
        "HouseAge":   house_age,
        "AveRooms":   ave_rooms,
        "AveBedrms":  ave_bedrms,
        "Population": population,
        "AveOccup":   ave_occup,
        "Latitude":   latitude,
        "Longitude":  longitude,
    }])

    prediction = float(model.predict(input_df)[0])
    usd_value  = prediction * 100_000
    rmse       = metrics["rmse"]
    low        = max((prediction - rmse) * 100_000, 0)
    high       = (prediction + rmse) * 100_000

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Estimated Median House Value", f"${usd_value:,.0f}")
    with res_col2:
        st.metric("Raw Model Output", f"{prediction:.4f} × $100k")
    with res_col3:
        st.metric("Approx. Range (±1 RMSE)", f"${low:,.0f} – ${high:,.0f}")

    st.info(
        f"The model predicts a median house value of **${usd_value:,.0f}** "
        f"for a district with these characteristics. "
        f"A realistic range is **${low:,.0f} – ${high:,.0f}** (±1 RMSE)."
    )