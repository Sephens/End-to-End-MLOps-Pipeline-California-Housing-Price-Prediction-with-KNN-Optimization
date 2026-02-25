"""
Training script for California Housing Price Prediction.

Loads data, builds the KNN pipeline, tunes hyperparameters via
GridSearchCV, evaluates on the test set, and persists the best model.

Usage:
    python pipeline/train.py
"""

import pickle
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

from pipeline.preprocess import build_preprocessor
from pipeline.evaluate import evaluate_model

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "california_knn_pipeline.pkl"
TEST_SIZE   = 0.2
RANDOM_STATE = 42

PARAM_GRID = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights":     ["uniform", "distance"],
    "knn__p":           [1, 2],
}


def load_data():
    """Fetch and return (X, y) as pandas DataFrames."""
    housing = fetch_california_housing(return_X_y=True, as_frame=True)
    X, y = housing
    print(f"Dataset loaded — shape: {X.shape}")
    return X, y


def build_pipeline() -> Pipeline:
    """Assemble preprocessing + KNN estimator into a single Pipeline."""
    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("knn", KNeighborsRegressor()),
        ]
    )
    return pipeline


def train(X_train, y_train) -> GridSearchCV:
    """Run GridSearchCV and return the fitted search object."""
    pipeline = build_pipeline()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="r2",
        verbose=1,
        n_jobs=-1,
    )

    print("\nStarting GridSearchCV …")
    grid_search.fit(X_train, y_train)
    print(f"\nBest params : {grid_search.best_params_}")
    print(f"Best CV R²  : {grid_search.best_score_:.4f}")

    return grid_search


def save_model(model, path: str = MODEL_PATH):
    """Persist the fitted pipeline to disk with pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n📦 Model saved to '{path}'")


def main():
    mlflow.set_experiment("california-housing-knn")

    with mlflow.start_run():
        # ── Data ──────────────────────────────────────────────────────────────
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # ── Train ─────────────────────────────────────────────────────────────
        grid_search  = train(X_train, y_train)
        best_model   = grid_search.best_estimator_

        # ── Evaluate ──────────────────────────────────────────────────────────
        metrics = evaluate_model(best_model, X_test, y_test)

        # ── Log to MLflow ─────────────────────────────────────────────────────
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")

        # ── Persist ───────────────────────────────────────────────────────────
        save_model(best_model)

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
