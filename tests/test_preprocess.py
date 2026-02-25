"""
Unit tests for pipeline/preprocess.py
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.preprocess import build_preprocessor, FEATURE_COLUMNS


@pytest.fixture
def sample_df():
    """A small clean DataFrame with all required features."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame(
        {
            "MedInc":     np.random.uniform(0.5,  15.0, n),
            "HouseAge":   np.random.uniform(1,    52.0, n),
            "AveRooms":   np.random.uniform(1.0,  20.0, n),
            "AveBedrms":  np.random.uniform(0.5,   5.0, n),
            "Population": np.random.uniform(3,  3500.0, n),
            "AveOccup":   np.random.uniform(1.0,  10.0, n),
            "Latitude":   np.random.uniform(32.5, 42.0, n),
            "Longitude":  np.random.uniform(-124.5, -114.3, n),
        }
    )


@pytest.fixture
def df_with_missing(sample_df):
    """Same DataFrame with a few NaN values injected."""
    df = sample_df.copy()
    df.loc[0, "MedInc"]     = np.nan
    df.loc[5, "HouseAge"]   = np.nan
    df.loc[10, "AveRooms"]  = np.nan
    return df


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_output_shape(self, sample_df):
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(sample_df)
        assert result.shape == (len(sample_df), len(FEATURE_COLUMNS))

    def test_imputes_missing_values(self, df_with_missing):
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(df_with_missing)
        assert not np.isnan(result).any(), "Output should have no NaN values after imputation"

    def test_scales_to_zero_mean(self, sample_df):
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(sample_df)
        col_means = result.mean(axis=0)
        np.testing.assert_allclose(col_means, 0, atol=1e-8)

    def test_scales_to_unit_variance(self, sample_df):
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(sample_df)
        col_stds = result.std(axis=0)
        np.testing.assert_allclose(col_stds, 1, atol=1e-8)

    def test_feature_columns_constant(self):
        assert len(FEATURE_COLUMNS) == 8
        assert "MedInc" in FEATURE_COLUMNS
        assert "Latitude" in FEATURE_COLUMNS
        assert "Longitude" in FEATURE_COLUMNS
