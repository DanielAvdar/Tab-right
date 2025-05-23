"""Tests for the DriftCalculator implementation."""

import numpy as np
import pandas as pd
import pytest

from tab_right.drift.drift_calculator import DriftCalculator


@pytest.fixture
def sample_data():
    """Provides two sample dataframes for testing."""
    df1 = pd.DataFrame({
        "numeric_stable": np.arange(100),
        "numeric_drifted": np.arange(100) + 10,
        "category_stable": np.random.choice(["A", "B", "C"], 100),
        "category_drifted": np.random.choice(["A", "B", "D"], 100),
        "low_card_numeric": np.random.choice([1, 2, 3], 100),
    })
    df2 = pd.DataFrame({
        "numeric_stable": np.arange(100),
        "numeric_drifted": np.arange(100) + 20,
        "category_stable": np.random.choice(["A", "B", "C"], 100),
        "category_drifted": np.random.choice(["A", "B", "D"], 100),
        "low_card_numeric": np.random.choice([1, 2, 4], 100),
    })
    return df1, df2


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
def test_feature_types_and_call(sample_data, col, expected_type):
    df1, df2 = sample_data
    calc = DriftCalculator(df1, df2)
    assert calc._feature_types[col] == expected_type
    res = calc()
    assert col in res.feature.values


@pytest.mark.parametrize(
    "kind,err_type",
    [([True], TypeError), (object(), TypeError), (None, None)],
)
def test_invalid_kind_and_no_common_columns(kind, err_type):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    if err_type:
        with pytest.raises(err_type):
            DriftCalculator(df, df, kind=kind)
    else:
        # No error for kind=None
        DriftCalculator(df, df, kind=kind)
    # No common columns
    with pytest.raises(ValueError):
        DriftCalculator(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]}))


def test_missing_kind_dict_entry(sample_data):
    df1, df2 = sample_data
    kind = {"numeric_stable": "continuous"}  # missing other columns
    calc = DriftCalculator(df1, df2, kind=kind)
    # Should fallback to auto detection for missing keys
    assert calc._feature_types["numeric_stable"] == "continuous"
    assert calc._feature_types["category_stable"] == "categorical"


def test_handle_missing_data(sample_data):
    df1, df2 = sample_data
    df1.loc[0, "numeric_stable"] = None  # Introduce missing value
    df2.loc[0, "numeric_stable"] = None

    results = DriftCalculator(df1, df2)()
    assert not results.empty
    assert "numeric_stable" in results.feature.values
    assert results.loc[results.feature == "numeric_stable", "score"].iloc[0] >= 0


def test_unknown_feature_type(sample_data):
    df1, df2 = sample_data
    calc = DriftCalculator(df1, df2)
    calc._feature_types["numeric_stable"] = "unknown"
    with pytest.raises(ValueError):
        calc(["numeric_stable"])
