"""Tests for the DriftCalculator implementation."""

import pandas as pd
import pytest

from tab_right.drift.drift_calculator import DriftCalculator


def make_df(cols: list, vals: list) -> pd.DataFrame:
    return pd.DataFrame({c: v for c, v in zip(cols, vals)})


@pytest.mark.parametrize(
    "df1,df2,kind,expected_types,expect_error,prob_density_type",
    [
        # Standard types (a: numeric with 3 unique values, so categorical)
        (
            make_df(["a", "b"], [[1, 2, 3], ["x", "y", "z"]]),
            make_df(["a", "b"], [[2, 3, 4], ["x", "y", "z"]]),
            None,
            {"a": "categorical", "b": "categorical"},
            None,
            "categorical",
        ),
        # Kind dict fallback
        (
            make_df(["a", "b"], [[1, 2, 3], ["x", "y", "z"]]),
            make_df(["a", "b"], [[2, 3, 4], ["x", "y", "z"]]),
            {"a": "continuous"},
            {"a": "continuous", "b": "categorical"},
            None,
            "continuous",
        ),
        # Invalid kind type
        (
            make_df(["a"], [[1, 2]]),
            make_df(["a"], [[1, 2]]),
            123,
            None,
            TypeError,
            None,
        ),
        # No common columns
        (
            make_df(["a"], [[1]]),
            make_df(["b"], [[2]]),
            None,
            None,
            ValueError,
            None,
        ),
    ],
)
def test_drift_calculator_all(df1, df2, kind, expected_types, expect_error, prob_density_type):
    if expect_error:
        with pytest.raises(expect_error):
            DriftCalculator(df1, df2, kind=kind)
    else:
        calc = DriftCalculator(df1, df2, kind=kind)
        for col, typ in expected_types.items():
            assert calc._feature_types[col] == typ
        res = calc()
        for col in expected_types:
            assert col in res.feature.values
        # get_prob_density for at least one type
        if prob_density_type:
            if prob_density_type == "continuous":
                calc._feature_types[list(expected_types.keys())[0]] = "continuous"
            dens = calc.get_prob_density()
            assert "feature" in dens.columns and "bin" in dens.columns


@pytest.mark.parametrize(
    "series1,series2,expected_cat,expected_cont",
    [
        (pd.Series([], dtype=object), pd.Series([], dtype=object), 0.0, None),
        (pd.Series(["a", "b"]), pd.Series(["a", "b"]), 0.0, None),
        (pd.Series([1.0, 2.0]), pd.Series([]), None, 1.0),
        (pd.Series([]), pd.Series([1.0, 2.0]), None, 1.0),
        (pd.Series([], dtype=float), pd.Series([], dtype=float), None, 0.0),
    ],
)
def test_drift_calculator_micro(series1, series2, expected_cat, expected_cont):
    from tab_right.drift.drift_calculator import DriftCalculator

    if expected_cat is not None:
        assert DriftCalculator._categorical_drift_calc(series1, series2) == expected_cat
    if expected_cont is not None:
        assert DriftCalculator._continuous_drift_calc(series1, series2) == expected_cont


def test_drift_calculator_unknown_feature_type():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [2, 3, 4]})
    calc = DriftCalculator(df1, df2)
    calc._feature_types["a"] = "unknown"
    with pytest.raises(ValueError):
        calc(["a"])


def test_drift_calculator_get_prob_density_skip_unknown():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [2, 3, 4]})
    calc = DriftCalculator(df1, df2)
    calc._feature_types["a"] = "unknown"
    with pytest.raises(ValueError):
        calc.get_prob_density()
