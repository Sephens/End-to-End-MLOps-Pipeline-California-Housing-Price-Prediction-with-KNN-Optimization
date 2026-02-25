"""
Preprocessing pipeline for California Housing dataset.
Handles imputation and feature scaling via Scikit-Learn Pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Expected feature columns (matches California Housing dataset)
FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return the preprocessing ColumnTransformer.

    Steps:
        1. SimpleImputer  — fills missing values with column mean
        2. StandardScaler — zero mean, unit variance

    Returns:
        ColumnTransformer configured for all numeric features.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS),
        ],
        remainder="drop",
    )

    return preprocessor
