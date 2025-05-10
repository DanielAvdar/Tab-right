"""Tests for the DriftCalculator implementation."""

import numpy as np
import pandas as pd
import pytest

from tab_right.drift.drift_calculator import DriftCalculator


@pytest.fixture
def sample_data():
    """Provides two sample dataframes for testing."""
    df1 = pd.DataFrame({
        "numeric_stable": np.random.normal(0, 1, 1000),
        "numeric_drifted": np.random.normal(0, 1, 1000),
        "category_stable": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]),
        "category_drifted": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]),
        "low_card_numeric": np.random.choice([1, 2, 3], 1000),
    })
    df2 = pd.DataFrame({
        "numeric_stable": np.random.normal(0, 1, 1100),  # Slightly different size
        "numeric_drifted": np.random.normal(2, 1.5, 1100),  # Mean and std dev drift
        "category_stable": np.random.choice(["A", "B", "C"], 1100, p=[0.5, 0.3, 0.2]),
        "category_drifted": np.random.choice(
            ["A", "B", "D"], 1100, p=[0.2, 0.3, 0.5]
        ),  # Category proportions and values drift
        "low_card_numeric": np.random.choice([1, 2, 4], 1100),  # Value drift
    })
    return df1, df2


def test_drift_calculator_init(sample_data):
    """Test initialization and automatic type detection."""
    df1, df2 = sample_data
    calculator = DriftCalculator(df1, df2, kind="auto")
    assert isinstance(calculator, DriftCalculator)
    assert "numeric_stable" in calculator._feature_types
    assert calculator._feature_types["numeric_stable"] == "continuous"
    assert calculator._feature_types["numeric_drifted"] == "continuous"
    assert calculator._feature_types["category_stable"] == "categorical"
    assert calculator._feature_types["category_drifted"] == "categorical"
    # Low cardinality numeric should be treated as categorical by default heuristic
    assert calculator._feature_types["low_card_numeric"] == "categorical"


def test_drift_calculator_call(sample_data):
    """Test the __call__ method for calculating drift scores."""
    df1, df2 = sample_data
    calculator = DriftCalculator(df1, df2)
    results = calculator()

    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == ["feature", "type", "score", "raw_score"]
    assert len(results) == 5  # Number of common columns

    # Check types assigned
    assert results.loc[results["feature"] == "numeric_stable", "type"].iloc[0] == "wasserstein"
    assert results.loc[results["feature"] == "category_stable", "type"].iloc[0] == "cramer_v"

    # Basic sanity checks on scores (exact values depend on random data)
    stable_num_score = results.loc[results["feature"] == "numeric_stable", "score"].iloc[0]
    drifted_num_score = results.loc[results["feature"] == "numeric_drifted", "score"].iloc[0]
    stable_cat_score = results.loc[results["feature"] == "category_stable", "score"].iloc[0]
    drifted_cat_score = results.loc[results["feature"] == "category_drifted", "score"].iloc[0]
    low_card_score = results.loc[results["feature"] == "low_card_numeric", "score"].iloc[0]

    assert drifted_num_score > stable_num_score
    assert drifted_cat_score > stable_cat_score
    assert low_card_score > 0  # Should detect some drift
    assert 0 <= stable_cat_score <= 1  # Cramer's V is normalized
    assert 0 <= drifted_cat_score <= 1


def test_get_prob_density(sample_data):
    """Test the get_prob_density method."""
    df1, df2 = sample_data
    calculator = DriftCalculator(df1, df2)
    densities = calculator.get_prob_density()

    assert isinstance(densities, pd.DataFrame)
    assert list(densities.columns) == ["feature", "bin", "ref_density", "cur_density"]
    assert set(densities["feature"].unique()) == set(df1.columns)

    # Check densities sum close to 1 for each feature
    for feature in df1.columns:
        feature_densities = densities[densities["feature"] == feature]
        assert np.isclose(feature_densities["ref_density"].sum(), 1.0, atol=1e-6)
        assert np.isclose(feature_densities["cur_density"].sum(), 1.0, atol=1e-6)

    # Check categorical bins
    cat_density = densities[densities["feature"] == "category_stable"]
    assert set(cat_density["bin"]) == {"A", "B", "C"}

    # Check continuous bins (default 10 bins)
    num_density = densities[densities["feature"] == "numeric_stable"]
    assert len(num_density["bin"]) == 10
    assert "(" in num_density["bin"].iloc[0]  # Check bin format


def test_categorical_drift_calc():
    """Test _categorical_drift_calc directly."""
    s1 = pd.Series(["A"] * 50 + ["B"] * 50)
    s2 = pd.Series(["A"] * 50 + ["B"] * 50)  # Identical
    s3 = pd.Series(["A"] * 20 + ["B"] * 80)  # Different proportions
    s4 = pd.Series(["C"] * 50 + ["D"] * 50)  # Different categories
    s5 = pd.Series(["A"] * 100)  # Single category
    s6 = pd.Series(["B"] * 100)  # Different single category

    assert np.isclose(DriftCalculator._categorical_drift_calc(s1, s2), 0.0)
    assert DriftCalculator._categorical_drift_calc(s1, s3) > 0.1  # Should be some drift
    assert np.isclose(DriftCalculator._categorical_drift_calc(s1, s4), 1.0)  # Max drift (no overlap)
    # Check for significant drift with single category comparison
    drift_score = DriftCalculator._categorical_drift_calc(s1, s5)
    assert 0 <= drift_score <= 1.0, "Drift score should be between 0 and 1"
    assert drift_score > 0.5  # Check for significant drift
    assert np.isclose(DriftCalculator._categorical_drift_calc(s5, s5), 0.0)  # Identical categories
    # Different single categories with floating point tolerance
    assert np.isclose(DriftCalculator._categorical_drift_calc(s5, s6), 1.0, atol=1e-2)


def test_continuous_drift_calc():
    """Test _continuous_drift_calc directly."""
    s1 = pd.Series(np.random.normal(0, 1, 500))
    s2 = pd.Series(np.random.normal(0, 1, 500))  # Similar distribution
    s3 = pd.Series(np.random.normal(5, 1, 500))  # Different mean
    s4 = pd.Series(np.random.normal(0, 3, 500))  # Different std dev

    # Wasserstein distance is non-negative
    assert DriftCalculator._continuous_drift_calc(s1, s1) == 0.0
    assert DriftCalculator._continuous_drift_calc(s1, s2) >= 0
    assert DriftCalculator._continuous_drift_calc(s1, s3) > DriftCalculator._continuous_drift_calc(s1, s2)
    assert DriftCalculator._continuous_drift_calc(s1, s4) > DriftCalculator._continuous_drift_calc(s1, s2)


def test_handle_missing_data(sample_data):
    """Test DriftCalculator handles missing data gracefully."""
    df1, df2 = sample_data
    df1.loc[0, "numeric_stable"] = None  # Introduce missing value
    df2.loc[0, "numeric_stable"] = None

    calculator = DriftCalculator(df1, df2)
    results = calculator()

    assert not results.empty
    assert "numeric_stable" in results["feature"].values
    assert results.loc[results["feature"] == "numeric_stable", "score"].iloc[0] >= 0


def test_invalid_input():
    """Test DriftCalculator raises errors for invalid inputs."""
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [4, 5, 6]})  # Different columns

    with pytest.raises(ValueError):
        DriftCalculator(df1, df2)
