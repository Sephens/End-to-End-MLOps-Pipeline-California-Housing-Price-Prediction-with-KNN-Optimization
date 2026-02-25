"""
Streamlit frontend for the California Housing Price Predictor.
Calls the deployed FastAPI /predict endpoint.
"""

import os
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CA Housing Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 California Housing Price Predictor")
st.markdown(
    "Adjust the sliders to describe a district, then click **Predict** to get "
    "the estimated median house value."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    med_inc    = st.slider("Median Income ($10k)",      0.5,  15.0,  3.5,  step=0.1)
    house_age  = st.slider("House Age (years)",          1.0,  52.0, 25.0,  step=1.0)
    ave_rooms  = st.slider("Avg Rooms / Household",      1.0,  20.0,  5.2,  step=0.1)
    ave_bedrms = st.slider("Avg Bedrooms / Household",   0.5,   5.0,  1.1,  step=0.1)

with col2:
    population = st.slider("Block Population",           3.0, 5000.0, 800.0, step=10.0)
    ave_occup  = st.slider("Avg Occupancy",              1.0,  10.0,  3.0,  step=0.1)
    latitude   = st.slider("Latitude",                  32.5,  42.0, 37.5,  step=0.1)
    longitude  = st.slider("Longitude",               -124.5,-114.3,-120.5,  step=0.1)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
    payload = {
        "MedInc":     med_inc,
        "HouseAge":   house_age,
        "AveRooms":   ave_rooms,
        "AveBedrms":  ave_bedrms,
        "Population": population,
        "AveOccup":   ave_occup,
        "Latitude":   latitude,
        "Longitude":  longitude,
    }

    with st.spinner("Calling prediction API…"):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.success("Prediction complete!")
            st.metric(
                label="Estimated Median House Value",
                value=result["predicted_value_usd"],
            )
            st.caption(
                f"Raw model output: {result['predicted_value_100k']} "
                f"(× $100,000) · Model v{result['model_version']}"
            )

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Could not connect to the API at `{API_URL}`. "
                "Make sure the FastAPI service is running and `API_URL` is set correctly."
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API returned an error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This app predicts **median house values** for California districts
        using a K-Nearest Neighbors model trained on the
        [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

        **Model performance**
        - R² score: `0.70`
        - Algorithm: KNN + GridSearchCV
        - CV folds: 5

        **Stack**
        - Frontend: Streamlit
        - Backend: FastAPI
        - ML: Scikit-Learn
        """
    )
    st.divider()
    st.caption(f"API endpoint: `{API_URL}`")
