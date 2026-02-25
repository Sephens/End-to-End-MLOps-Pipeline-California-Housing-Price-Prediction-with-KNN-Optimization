"""
Prefect orchestration flow for the California Housing training pipeline.

Each logical step is wrapped as a @task so Prefect can monitor,
retry, and visualise the workflow in its UI.

Usage:
    python flows/training_flow.py
    prefect deployment build flows/training_flow.py:training_flow -n "daily-retrain"
"""

import pickle

import mlflow
import mlflow.sklearn
from prefect import flow, task, get_run_logger
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

from pipeline.preprocess import build_preprocessor, FEATURE_COLUMNS
from pipeline.evaluate import evaluate_model

MODEL_PATH   = "california_knn_pipeline.pkl"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

PARAM_GRID = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights":     ["uniform", "distance"],
    "knn__p":           [1, 2],
}


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(name="Load Data", retries=2, retry_delay_seconds=5)
def load_data_task():
    logger = get_run_logger()
    housing = fetch_california_housing(return_X_y=True, as_frame=True)
    X, y = housing
    logger.info(f"Data loaded — {X.shape[0]:,} rows, {X.shape[1]} features")
    return X, y


@task(name="Split Data")
def split_data_task(X, y):
    logger = get_run_logger()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


@task(name="Train Model", timeout_seconds=600)
def train_model_task(X_train, y_train):
    logger = get_run_logger()

    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("knn", KNeighborsRegressor()),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="r2",
        verbose=0,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    logger.info(f"Best params : {grid_search.best_params_}")
    logger.info(f"Best CV R²  : {grid_search.best_score_:.4f}")
    return grid_search


@task(name="Evaluate Model")
def evaluate_model_task(grid_search, X_test, y_test):
    logger = get_run_logger()
    best_model = grid_search.best_estimator_
    metrics    = evaluate_model(best_model, X_test, y_test)
    logger.info(f"Test metrics: {metrics}")
    return best_model, metrics


@task(name="Log to MLflow")
def log_mlflow_task(grid_search, best_model, metrics):
    mlflow.set_experiment("california-housing-knn")
    with mlflow.start_run():
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")


@task(name="Save Model")
def save_model_task(best_model):
    logger = get_run_logger()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"Model saved to '{MODEL_PATH}'")


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(name="California Housing Training Pipeline")
def training_flow():
    X, y                               = load_data_task()
    X_train, X_test, y_train, y_test   = split_data_task(X, y)
    grid_search                        = train_model_task(X_train, y_train)
    best_model, metrics                = evaluate_model_task(grid_search, X_test, y_test)
    log_mlflow_task(grid_search, best_model, metrics)
    save_model_task(best_model)


if __name__ == "__main__":
    training_flow()
