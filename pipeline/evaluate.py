"""
Evaluation utilities for the California Housing model.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Run predictions and compute regression metrics.

    Args:
        model:   Fitted Scikit-Learn pipeline/estimator.
        X_test:  Test feature DataFrame.
        y_test:  True target values.

    Returns:
        Dictionary containing r2, mse, and rmse scores.
    """
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "r2":   round(r2,   4),
        "mse":  round(mse,  4),
        "rmse": round(rmse, 4),
    }

    print("\n=== Model Evaluation ===")
    print(f"  R²   : {metrics['r2']}")
    print(f"  MSE  : {metrics['mse']}")
    print(f"  RMSE : {metrics['rmse']}")

    return metrics
