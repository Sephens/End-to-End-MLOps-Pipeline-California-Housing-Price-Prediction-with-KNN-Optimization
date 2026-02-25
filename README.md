# 🏠 California Housing Price Prediction — End-to-End MLOps Pipeline

> An end-to-end ML pipeline predicting California housing prices using Scikit-Learn, served as a live REST API with full MLOps tooling.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-red?logo=mlflow)](https://mlflow.org/)

---

## 📋 Overview

This project demonstrates core **MLOps principles** by building a production-ready machine learning pipeline that predicts median house prices for California districts using census data. The model is not just a training script — it's a **deployed microservice** with automated workflows, experiment tracking, and containerized infrastructure.

**Model Performance:** `R² = 0.70` on held-out test data.

---

## ✨ Key Features

| Area | What Was Built |
|------|---------------|
| **Data** | California Housing dataset with imputation & feature scaling |
| **Modeling** | KNN Regressor with GridSearchCV hyperparameter tuning (5-fold CV) |
| **Deployment** | FastAPI REST endpoint (`/predict`) serving real-time predictions |
| **Containerization** | Dockerized microservice, tested on Google Cloud Run |
| **Orchestration** | Prefect flow automating the full training pipeline |
| **Tracking** | MLflow logging for parameters, metrics, and model artifacts |
| **Testing** | Pytest suite validating data processing logic and API endpoints |

---

## 🏗️ Architecture

```
Raw Data → Preprocessing Pipeline → KNN Model (GridSearchCV)
                                          ↓
                                   MLflow Tracking
                                          ↓
                               Prefect Orchestration
                                          ↓
                              FastAPI /predict Endpoint
                                          ↓
                                  Docker Container
                                          ↓
                               Google Cloud Run (Live)
```

---

## 🔬 ML Pipeline Details

**Preprocessing**
- Missing value imputation using column means
- StandardScaler normalization for all numerical features
- Scikit-Learn `Pipeline` + `ColumnTransformer` for clean, leak-free preprocessing

**Hyperparameter Tuning**
The following grid was searched using `GridSearchCV` with 5-fold cross-validation:

```python
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights':     ['uniform', 'distance'],
    'knn__p':           [1, 2]
}
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Docker
- Poetry

### Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/california-housing-mlops.git
cd california-housing-mlops

# Install dependencies
poetry install

# Train the model
python train.py

# Start the API
uvicorn app.main:app --reload
```

### Run with Docker

```bash
docker build -t california-housing-api .
docker run -p 8000:8000 california-housing-api
```

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 3.5,
    "HouseAge": 25,
    "AveRooms": 5.2,
    "AveBedrms": 1.1,
    "Population": 800,
    "AveOccup": 3.0,
    "Latitude": 37.5,
    "Longitude": -120.5
  }'
```

**Response:**
```json
{ "predicted_value": 2.41 }
```
> Value is in $100k units — e.g. `2.41` = **$241,000**

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Best CV R² Score | 0.72 |
| Test R² Score | **0.70** |
| Test MSE | ~0.48 |

---

## 🧱 MLOps Principles Demonstrated

**Reproducibility** — Docker image ensures identical environments across machines. Poetry pins all dependencies.

**Automation** — Prefect orchestrates every step (data loading → preprocessing → training → evaluation) as a managed, observable flow.

**Validation** — GridSearchCV with cross-validation during training; Pytest unit tests for processing logic and API endpoints.

**Deployment Readiness** — Model is served as a REST API, containerized, and deployed to Google Cloud Run.

**Tracking** — MLflow records all experiment parameters and metrics, enabling easy comparison across runs.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML | Scikit-Learn, Pandas, NumPy |
| API | FastAPI, Uvicorn |
| Containerization | Docker |
| Orchestration | Prefect |
| Experiment Tracking | MLflow |
| Testing | Pytest |
| Environment | Poetry |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
.
├── app/
│   └── main.py                  # FastAPI app
├── pipeline/
│   ├── train.py                 # Training script
│   ├── preprocess.py            # Preprocessing logic
│   └── evaluate.py              # Evaluation utilities
├── flows/
│   └── training_flow.py         # Prefect workflow
├── tests/
│   ├── test_preprocess.py
│   └── test_api.py
├── california_knn_pipeline.pkl  # Saved model artifact
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License.
