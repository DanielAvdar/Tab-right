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


@pytest.mark.parametrize(
    "col,expected_type",
    [
        ("numeric_stable", "continuous"),
        ("numeric_drifted", "continuous"),
        ("category_stable", "categorical"),
        ("category_drifted", "categorical"),
        ("low_card_numeric", "categorical"),
    ],
)
def test_feature_types(sample_data, col, expected_type):
    df1, df2 = sample_data
    calc = DriftCalculator(df1, df2, kind="auto")
    assert calc._feature_types[col] == expected_type


def test_drift_calculator_call_short(sample_data):
    df1, df2 = sample_data
    calc = DriftCalculator(df1, df2)
    res = calc()
    assert set(res.columns) == {"feature", "type", "score", "raw_score"}
    assert len(res) == 5
    # Types
    types = dict(zip(res.feature, res.type))
    assert types["numeric_stable"] == "wasserstein"
    assert types["category_stable"] == "cramer_v"
    # Score ranges
    for s in res.score:
        assert np.isnan(s) or 0 <= s <= 1 or s > 0  # Wasserstein can be >1


def test_get_prob_density(sample_data):
    """Test the get_prob_density method."""
    df1, df2 = sample_data
    calculator = DriftCalculator(df1, df2)
    densities = calculator.get_prob_density()

    assert isinstance(densities, pd.DataFrame)
    assert list(densities.columns) == ["feature", "bin", "ref_density", "cur_density"]
    assert set(densities["feature"].unique()) == set(df1.columns)

    # Densities sum to 1
    for f in df1.columns:
        d = densities[densities.feature == f]
        assert np.isclose(d.ref_density.sum(), 1.0, atol=1e-6)
        assert np.isclose(d.cur_density.sum(), 1.0, atol=1e-6)


def test_categorical_drift_calc():
    """Test _categorical_drift_calc directly."""
    s1 = pd.Series(["A"] * 50 + ["B"] * 50)
    s2 = pd.Series(["A"] * 50 + ["B"] * 50)  # Identical
    s3 = pd.Series(["A"] * 20 + ["B"] * 80)  # Different proportions
    s4 = pd.Series(["C"] * 50 + ["D"] * 50)  # Different categories
    s5 = pd.Series(["A"] * 100)  # Single category
    s6 = pd.Series(["B"] * 100)  # Different single category

    # Identical
    assert DriftCalculator._categorical_drift_calc(s1, s2) == 0.0
    # Different proportions
    assert DriftCalculator._categorical_drift_calc(s1, s3) > 0.1
    # No overlap
    assert DriftCalculator._categorical_drift_calc(s1, s4) == 1.0
    # Single category vs different single
    # Allow for floating point tolerance
    assert np.isclose(DriftCalculator._categorical_drift_calc(s5, s6), 1.0, atol=0.02)
    # Single category vs itself
    assert DriftCalculator._categorical_drift_calc(s5, s5) == 0.0


def test_continuous_drift_calc():
    """Test _continuous_drift_calc directly."""
    s1 = pd.Series(np.random.normal(0, 1, 500))
    s2 = pd.Series(np.random.normal(0, 1, 500))  # Similar distribution
    s3 = pd.Series(np.random.normal(5, 1, 500))  # Different mean
    s4 = pd.Series(np.random.normal(0, 3, 500))  # Different std dev

    # Wasserstein distance is non-negative and 0 for identical
    assert DriftCalculator._continuous_drift_calc(s1, s1) == 0.0
    assert DriftCalculator._continuous_drift_calc(s1, s2) >= 0
    assert DriftCalculator._continuous_drift_calc(s1, s3) > 0
    assert DriftCalculator._continuous_drift_calc(s1, s4) > 0


def test_handle_missing_data(sample_data):
    """Test DriftCalculator handles missing data gracefully."""
    df1, df2 = sample_data
    df1.loc[0, "numeric_stable"] = None  # Introduce missing value
    df2.loc[0, "numeric_stable"] = None

    results = DriftCalculator(df1, df2)()
    assert not results.empty
    assert "numeric_stable" in results.feature.values
    assert results.loc[results.feature == "numeric_stable", "score"].iloc[0] >= 0


def test_invalid_input():
    """Test DriftCalculator raises errors for invalid inputs."""
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [4, 5, 6]})  # Different columns

    with pytest.raises(ValueError):
        DriftCalculator(df1, df2)


# Additional coverage for error/edge cases in drift_calculator


def test_drift_calculator_edge_cases():
    # No common columns
    with pytest.raises(ValueError):
        DriftCalculator(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]}))
    # Kind iterable wrong length
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        DriftCalculator(df, df, kind=[True])
    # Kind wrong type
    with pytest.raises(TypeError):
        DriftCalculator(df, df, kind=object())
    # (moved unknown column type test to its own function)


# Test unknown column type in __call__
def test_drift_calculator_unknown_col_type():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    calc = DriftCalculator(df, df, kind="auto")
    # forcibly set an unknown type after construction
    calc._feature_types["a"] = "unknown"
    with pytest.raises(ValueError):
        calc(["a"])  # should raise
    # _categorical_drift_calc: empty series
    assert DriftCalculator._categorical_drift_calc(pd.Series([], dtype=object), pd.Series([], dtype=object)) == 0.0
    # _continuous_drift_calc: one empty
    assert DriftCalculator._continuous_drift_calc(pd.Series([], dtype=float), pd.Series([1.0])) == 1.0
