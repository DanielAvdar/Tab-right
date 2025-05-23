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


def test_normalization_zero_variance():
    from tab_right.drift.univariate import normalize_wasserstein

    s = pd.Series([1, 1, 1, 1])
    # std normalization with zero variance
    assert normalize_wasserstein(s, s, 1.0, method="std") == 0.0
    # iqr normalization with zero variance
    assert normalize_wasserstein(s, s, 1.0, method="iqr") == 0.0


def test_normalize_wasserstein_zero_value():
    from tab_right.drift.univariate import normalize_wasserstein

    s = pd.Series([1, 2, 3])
    # Should return 0.0 if wasserstein_value == 0
    assert normalize_wasserstein(s, s, 0.0, method="range") == 0.0


def test_univariate_all_nan_column():
    import pandas as pd

    from tab_right.drift.univariate import UnivariateDriftCalculator

    df1 = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    df2 = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    calc = UnivariateDriftCalculator(df1, df2, kind=None)
    with pytest.raises(ValueError):
        calc()


def test_univariate_invalid_kind_type():
    from tab_right.drift.univariate import UnivariateDriftCalculator

    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        UnivariateDriftCalculator(df1, df2, kind=123)


def test_univariate_kind_dict_missing_column():
    from tab_right.drift.univariate import UnivariateDriftCalculator

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    kind = {"a": "continuous"}  # missing 'b'
    calc = UnivariateDriftCalculator(df1, df2, kind=kind)
    result = calc()
    assert set(result["feature"]) == {"a", "b"}
    assert "cramer_v" in set(result["type"])  # fallback to auto for 'b'


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


def test_univariate_kind_dict_fallback_branches():
    # Covers fallback to categorical for non-numeric, and continuous for numeric with nunique > 20
    df1 = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "b": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x", "y", "z", "x", "y", "z", "x", "y", "z", "x", "y", "z"],
    })
    df2 = df1.copy()
    kind = {"b": "categorical"}  # missing 'a', which is numeric with nunique > 20
    calc = UnivariateDriftCalculator(df1, df2, kind=kind)
    result = calc()
    assert set(result["feature"]) == {"a", "b"}
    assert "wasserstein" in set(result["type"])  # fallback to continuous for 'a'
    assert "cramer_v" in set(result["type"])  # 'b' is categorical


def test_univariate_kind_dict_fallback_to_categorical():
    # 'b' is a string column, not in kind dict, should fallback to categorical
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    kind = {"a": "continuous"}  # missing 'b'
    calc = UnivariateDriftCalculator(df1, df2, kind=kind)
    result = calc()
    assert set(result["feature"]) == {"a", "b"}
    # 'b' should be categorical (cramer_v)
    assert result[result["feature"] == "b"]["type"].iloc[0] == "cramer_v"
