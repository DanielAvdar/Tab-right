import numpy as np
import pandas as pd
import pytest

from tab_right.drift.univariate import (
    UnivariateDriftCalculator,
    detect_univariate_drift,
    detect_univariate_drift_df,
    detect_univariate_drift_with_options,
)


@pytest.mark.parametrize(
    "ref, cur, kind, normalize, normalization_method, expected_type, expect_error, expect_nan",
    [
        # Valid continuous
        (pd.Series([1, 2, 3]), pd.Series([2, 3, 4]), "continuous", True, "range", "wasserstein", None, False),
        # Valid categorical
        (pd.Series(["a", "b", "a"]), pd.Series(["a", "b", "b"]), "categorical", True, "range", "cramer_v", None, False),
        # Auto detection (numeric)
        (pd.Series([1, 2, 3]), pd.Series([2, 3, 4]), "auto", True, "range", "wasserstein", None, False),
        # Auto detection (categorical)
        (pd.Series(["a", "b", "a"]), pd.Series(["a", "b", "b"]), "auto", True, "range", "cramer_v", None, False),
        # Invalid kind
        (pd.Series([1, 2, 3]), pd.Series([2, 3, 4]), "bad_kind", True, "range", None, ValueError, False),
        # Invalid normalization method
        (pd.Series([1, 2, 3]), pd.Series([2, 3, 4]), "continuous", True, "bad_method", None, ValueError, False),
        # Empty series
        (pd.Series([], dtype=float), pd.Series([], dtype=float), "continuous", True, "range", None, ValueError, False),
        # All NaN
        (
            pd.Series([np.nan, np.nan]),
            pd.Series([np.nan, np.nan]),
            "continuous",
            True,
            "range",
            "wasserstein",
            None,
            True,
        ),
    ],
)
def test_univariate_drift_cases(
    ref, cur, kind, normalize, normalization_method, expected_type, expect_error, expect_nan
):
    if expect_error:
        with pytest.raises(expect_error):
            detect_univariate_drift_with_options(
                ref, cur, kind=kind, normalize=normalize, normalization_method=normalization_method
            )
    else:
        result = detect_univariate_drift_with_options(
            ref, cur, kind=kind, normalize=normalize, normalization_method=normalization_method
        )
        assert result["type"] == expected_type
        if expect_nan:
            assert np.isnan(result["score"]) or pd.isna(result["score"])
        else:
            assert isinstance(result["score"], float)


@pytest.mark.parametrize("method", ["range", "std", "iqr"])
def test_normalization_methods(method):
    ref = pd.Series([1000, 2000, 3000, 4000, 5000])
    cur = pd.Series([2000, 3000, 4000, 5000, 6000])
    _, value = detect_univariate_drift(ref, cur, normalize=True, normalization_method=method)
    assert 0 <= value <= 1


def test_univariate_drift_df_and_types():
    df1 = pd.DataFrame({"num": list(range(25)), "cat": ["a", "b", "a", "b", "c"] * 5})
    df2 = pd.DataFrame({"num": list(range(1, 26)), "cat": ["a", "b", "b", "b", "c"] * 5})
    result = detect_univariate_drift_df(df1, df2)
    assert set(result["feature"]) == {"num", "cat"}
    assert all(m in ("wasserstein", "cramer_v") for m in result["metric"])
    # Normalized values should be between 0 and 1 for continuous
    for _, row in result.iterrows():
        if row["metric"] == "wasserstein":
            assert 0 <= row["value"] <= 1


def test_univariate_drift_calculator_kind_dict():
    df1 = pd.DataFrame({"num": list(range(25)), "cat": ["a", "b", "a", "b", "c"] * 5})
    df2 = pd.DataFrame({"num": list(range(1, 26)), "cat": ["a", "b", "b", "b", "c"] * 5})
    kind = {"num": "continuous", "cat": "categorical"}
    calc = UnivariateDriftCalculator(df1, df2, kind=kind)
    result = calc()
    assert set(result["feature"]) == {"num", "cat"}
    assert set(result["type"]) == {"wasserstein", "cramer_v"}
