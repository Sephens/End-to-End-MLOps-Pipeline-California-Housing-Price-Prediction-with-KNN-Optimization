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
    """Load model from disk if it exists, otherwise train and save it."""

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model, metrics = pickle.load(f)
        return model, metrics

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
        "r2":          round(r2_score(y_test, y_pred), 4),
        "rmse":        round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "best_params": grid.best_params_,
        "cv_r2":       round(grid.best_score_, 4),
    }

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((best_model, metrics), f)

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

    with st.spinner("Loading model (first run may take ~2 mins to train)…"):
        model, metrics = load_or_train_model()

    st.success("Model ready!")
    st.metric("Test R²",   metrics["r2"])
    st.metric("Test RMSE", f"{metrics['rmse']} ($100k)")
    st.metric("CV R²",     metrics["cv_r2"])

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
    med_inc    = st.slider("Median Income",             0.5,  15.0,  3.5,  step=0.1,
                           help="Median household income in the block group (in $10,000s)")
    house_age  = st.slider("House Age (years)",          1.0,  52.0, 25.0,  step=1.0,
                           help="Median age of houses in the block group")
    ave_rooms  = st.slider("Avg Rooms / Household",      1.0,  20.0,  5.2,  step=0.1)
    ave_bedrms = st.slider("Avg Bedrooms / Household",   0.5,   5.0,  1.1,  step=0.1)

with col2:
    st.subheader("👥 Population & Location")
    population = st.slider("Block Population",           3.0, 5000.0, 800.0, step=10.0)
    ave_occup  = st.slider("Avg Occupancy",              1.0,  10.0,  3.0,  step=0.1,
                           help="Average number of people per household")
    latitude   = st.slider("Latitude",                  32.5,  42.0, 37.5,  step=0.1,
                           help="California ranges from ~32.5°N to ~42°N")
    longitude  = st.slider("Longitude",               -124.5,-114.3,-120.5,  step=0.1,
                           help="California ranges from ~-124.5° to ~-114.3°")

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

    # Result display
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Estimated Median House Value", f"${usd_value:,.0f}")
    with res_col2:
        st.metric("Raw Model Output", f"{prediction:.4f} × $100k")
    with res_col3:
        # Rough confidence band: ±1 RMSE
        low  = (prediction - metrics["rmse"]) * 100_000
        high = (prediction + metrics["rmse"]) * 100_000
        st.metric("Approx. Range (±1 RMSE)", f"${max(low,0):,.0f} – ${high:,.0f}")

    st.info(
        f"**How to read this:** The model predicts a median house value of "
        f"**${usd_value:,.0f}** for a district with these characteristics. "
        f"Based on the model's RMSE, a realistic range is "
        f"**${max(low,0):,.0f} – ${high:,.0f}**."
    )