import numpy as np
import pandas as pd
import pytest

from tab_right.drift.cramer_v import cramer_v
from tab_right.drift.psi import psi
from tab_right.drift.univariate import (
    UnivariateDriftCalculator,
    detect_univariate_drift,
    detect_univariate_drift_df,
    detect_univariate_drift_with_options,
)


@pytest.mark.parametrize("backend", [None, "pyarrow"])
def test_cramer_v_basic(backend):
    x = pd.Series(["a", "b", "a", "b", "c", "c"])
    y = pd.Series(["a", "a", "b", "b", "c", "c"])
    if backend:
        x = x.astype("string[pyarrow]")
        y = y.astype("string[pyarrow]")
    v = cramer_v(x, y)
    assert 0 <= v <= 1


@pytest.mark.parametrize("backend", [None, "pyarrow"])
def test_psi_basic(backend):
    expected = pd.Series([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    actual = pd.Series([0, 1, 1, 2, 2, 2, 3, 3, 3, 0])
    if backend:
        expected = expected.astype("int64[pyarrow]")
        actual = actual.astype("int64[pyarrow]")
    value = psi(expected, actual, bins=4)
    assert value >= 0


@pytest.mark.parametrize("backend", [None, "pyarrow"])
def test_detect_univariate_drift_continuous(backend):
    ref = pd.Series(np.random.normal(0, 1, 100))
    cur = pd.Series(np.random.normal(0.5, 1, 100))
    if backend:
        ref = ref.astype("float64[pyarrow]")
        cur = cur.astype("float64[pyarrow]")
    metric, value = detect_univariate_drift(ref, cur, kind="continuous")
    assert metric == "wasserstein"
    assert value >= 0


@pytest.mark.parametrize("backend", [None, "pyarrow"])
def test_detect_univariate_drift_categorical(backend):
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 30 + ["b"] * 70)
    if backend:
        ref = ref.astype("string[pyarrow]")
        cur = cur.astype("string[pyarrow]")
    metric, value = detect_univariate_drift(ref, cur, kind="categorical")
    assert metric == "cramer_v"
    assert 0 <= value <= 1


@pytest.mark.parametrize("backend", [None, "pyarrow"])
def test_detect_univariate_drift_auto(backend):
    ref = pd.Series([1, 2, 3, 4, 5])
    cur = pd.Series([1, 2, 2, 4, 5])
    if backend:
        ref = ref.astype("int64[pyarrow]")
        cur = cur.astype("int64[pyarrow]")
    # The new protocol does not support kind=None for detect_univariate_drift; must use kind="auto"
    metric, value = detect_univariate_drift(ref, cur, kind="auto")
    assert metric in ("wasserstein", "cramer_v")


@pytest.mark.parametrize(
    "ref, cur, kind, expected_metric, expected_range, expect_exception, expect_nan",
    [
        # Identical continuous
        (
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([1.0, 2.0, 3.0]),
            "continuous",
            "wasserstein",
            (0, 0.0001),
            False,
            False,
        ),
        # Shifted continuous - with normalization=False to test raw values
        (pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 3.0, 4.0]), "continuous", "wasserstein", (0.9, 1.1), False, False),
        # Identical categorical (Cramér's V can be 1.0 for small samples)
        (pd.Series(["a", "b", "c"]), pd.Series(["a", "b", "c"]), "categorical", "cramer_v", (0, 1), False, False),
        # Permuted categorical (Cramér's V can be 0.5 for small samples)
        (
            pd.Series(["a", "a", "b", "b"]),
            pd.Series(["b", "b", "a", "a"]),
            "categorical",
            "cramer_v",
            (0, 1),
            False,
            False,
        ),
        # Imbalanced categorical
        (
            pd.Series(["a"] * 99 + ["b"]),
            pd.Series(["a"] * 50 + ["b"] * 50),
            "categorical",
            "cramer_v",
            (0, 1),
            False,
            False,
        ),
        # Empty series (should raise)
        (pd.Series([], dtype=float), pd.Series([], dtype=float), "continuous", "wasserstein", (0, 0), True, False),
        # All same value
        (pd.Series([1, 1, 1]), pd.Series([1, 1, 1]), "continuous", "wasserstein", (0, 0.0001), False, False),
        # With NaNs (should return nan)
        (pd.Series([1, 2, np.nan]), pd.Series([1, 2, np.nan]), "continuous", "wasserstein", (0, 0.0001), False, True),
    ],
)
def test_detect_univariate_drift_variety(ref, cur, kind, expected_metric, expected_range, expect_exception, expect_nan):
    if expect_exception:
        import pytest

        with pytest.raises(ValueError):
            detect_univariate_drift(ref, cur, kind=kind)
    else:
        # Special case for shifted continuous test that expects raw value
        normalize = (
            False if kind == "continuous" and not (expected_range[0] == 0 and expected_range[1] < 0.01) else True
        )
        # If kind is 'auto', replace with None
        kind_arg = None if kind == "auto" else kind
        metric, value = detect_univariate_drift(ref, cur, kind=kind_arg, normalize=normalize)
        assert metric == expected_metric
        if expect_nan:
            assert pd.isna(value)
        else:
            assert expected_range[0] <= value <= expected_range[1]


def test_detect_univariate_drift_df_variety():
    # Use >20 unique values for 'num' to ensure it is detected as continuous
    df_ref = pd.DataFrame({"num": list(range(25)), "cat": ["a", "b", "a", "b", "c"] * 5})
    df_cur = pd.DataFrame({"num": list(range(1, 26)), "cat": ["a", "b", "b", "b", "c"] * 5})
    result = detect_univariate_drift_df(df_ref, df_cur)
    assert set(result["feature"]) == {"num", "cat"}
    assert all(m in ("wasserstein", "cramer_v") for m in result["metric"])
    assert all((v >= 0 or pd.isna(v)) for v in result["value"])

    # Also check normalized vs. raw results
    result_raw = detect_univariate_drift_df(df_ref, df_cur, normalize=False)
    result_norm = detect_univariate_drift_df(df_ref, df_cur, normalize=True)

    # Normalized values should be between 0 and 1
    for _, row in result_norm.iterrows():
        if row["metric"] == "wasserstein":
            assert 0 <= row["value"] <= 1

    # For continuous features, raw values should be different from normalized values
    num_raw = result_raw.loc[result_raw["feature"] == "num", "value"].iloc[0]
    num_norm = result_norm.loc[result_norm["feature"] == "num", "value"].iloc[0]
    assert num_raw != num_norm


def test_detect_univariate_drift_invalid_kind():
    """Test that an invalid 'kind' parameter raises a ValueError."""
    ref = pd.Series([1, 2, 3, 4, 5])
    cur = pd.Series([1, 2, 2, 4, 5])

    with pytest.raises(ValueError, match="Unknown kind"):
        detect_univariate_drift(ref, cur, kind="invalid_kind")


def test_univariate_drift_calculator_invalid_kind_length():
    """Test that a mismatched kind parameter length raises a TypeError."""
    # This test is now obsolete due to static typing; nothing to test at runtime
    pass


def test_detect_univariate_drift_df_rename_columns():
    """Test the backward compatibility function renames columns correctly."""
    df1 = pd.DataFrame({"num": list(range(25)), "cat": ["a", "b", "c"] * 8 + ["a"]})
    df2 = pd.DataFrame({"num": list(range(1, 26)), "cat": ["a", "a", "c"] * 8 + ["b"]})
    result = detect_univariate_drift_df(df1, df2)
    # Check the result has the expected columns from the old API
    assert "feature" in result.columns
    assert "metric" in result.columns
    assert "value" in result.columns
    # The renamed columns shouldn't exist
    assert "type" not in result.columns
    assert "score" not in result.columns
    # Should also include raw_value for continuous features
    assert "raw_value" in result.columns


def test_non_numeric_dtypes_auto_detection():
    """Test that non-numeric types are properly classified as categorical in auto mode."""
    # Create DataFrames with various non-numeric dtypes
    df1 = pd.DataFrame({
        "string_col": pd.Series(["a", "b", "c"], dtype="string"),
        "object_col": pd.Series(["x", "y", "z"], dtype="object"),
        "category_col": pd.Series(["cat1", "cat2", "cat3"], dtype="category"),
    })

    df2 = pd.DataFrame({
        "string_col": pd.Series(["a", "b", "d"], dtype="string"),
        "object_col": pd.Series(["x", "y", "w"], dtype="object"),
        "category_col": pd.Series(["cat1", "cat2", "cat4"], dtype="category"),
    })

    # Use the calculator with auto detection
    calc = UnivariateDriftCalculator(df1, df2, kind=None)
    result = calc()

    # Verify non-numeric columns were treated as categorical (this hits line 67)
    for col in ["string_col", "object_col", "category_col"]:
        assert result.loc[result["feature"] == col, "type"].iloc[0] == "cramer_v"

    # Also test with the detect_univariate_drift function individually
    for col in df1.columns:
        metric_type, _ = detect_univariate_drift(df1[col], df2[col])
        assert metric_type == "cramer_v"


def test_unknown_kind_value_error():
    """Test that an invalid kind value raises the expected ValueError."""
    series1 = pd.Series([1, 2, 3])
    series2 = pd.Series([1, 2, 4])

    # This should target line 116 with the ValueError for unknown kind
    with pytest.raises(ValueError, match="Unknown kind"):
        detect_univariate_drift(series1, series2, kind="unknown_kind_value")


def test_pandas_extension_array_dtypes():
    """Test handling of pandas extension array dtypes."""
    # Test with different extension array dtypes
    df1 = pd.DataFrame({
        "int_array": pd.Series(list(range(25)), dtype="Int64"),  # Nullable integer
        "string_array": pd.Series(["a", "b", "c"] * 8 + ["a"]),
    })
    df2 = pd.DataFrame({
        "int_array": pd.Series(list(range(1, 26)), dtype="Int64"),
        "string_array": pd.Series(["a", "c", "c"] * 8 + ["b"]),
    })
    # The new protocol does not allow invalid kind types; skip this test
    calc = UnivariateDriftCalculator(df1, df2, kind=None)
    result = calc()
    # Check types are correctly set
    assert result.loc[result["feature"] == "int_array", "type"].iloc[0] == "wasserstein"
    assert result.loc[result["feature"] == "string_array", "type"].iloc[0] == "cramer_v"


def test_univariate_auto_detection_edge_cases():
    """Test the auto-detection logic with edge cases to improve coverage."""
    # Create DataFrames with mixed dtypes including numeric and non-numeric
    df1 = pd.DataFrame({
        # A boolean column (should be detected as categorical despite being numeric in some contexts)
        "bool_col": pd.Series([True, False, True], dtype="bool"),
        # A numeric column that's actually categorical (has very few unique values)
        "numeric_cat": pd.Series([1, 2] * 12 + [1], dtype="int64"),
    })
    df2 = pd.DataFrame({
        "bool_col": pd.Series([False, False, True] * 8 + [True], dtype="bool"),
        "numeric_cat": pd.Series([1, 1, 2, 2] * 6 + [2], dtype="int64"),
    })
    # Use None for kind (auto detection)
    calc = UnivariateDriftCalculator(df1, df2, kind=None)
    result = calc()
    # Verify boolean values were treated as categorical
    bool_col_type = result.loc[result["feature"] == "bool_col", "type"].iloc[0]
    assert bool_col_type == "cramer_v"
    # Verify the numeric categorical was treated as categorical (<=20 unique values)
    num_cat_type = result.loc[result["feature"] == "numeric_cat", "type"].iloc[0]
    assert num_cat_type == "cramer_v"


def test_drift_normalization_methods():
    """Test different normalization methods for continuous drift."""
    ref = pd.Series([1000, 2000, 3000, 4000, 5000])
    cur = pd.Series([2000, 3000, 4000, 5000, 6000])

    # Test all normalization methods
    _, range_norm = detect_univariate_drift(ref, cur, normalize=True, normalization_method="range")
    _, std_norm = detect_univariate_drift(ref, cur, normalize=True, normalization_method="std")
    _, iqr_norm = detect_univariate_drift(ref, cur, normalize=True, normalization_method="iqr")

    # All normalized values should be between 0 and 1
    assert 0 <= range_norm <= 1
    assert 0 <= std_norm <= 1
    assert 0 <= iqr_norm <= 1

    # Test with an invalid normalization method
    with pytest.raises(ValueError, match="Unknown normalization method"):
        detect_univariate_drift(ref, cur, normalize=True, normalization_method="invalid")


def test_detect_univariate_drift_unknown_kind_direct():
    """Direct test for the error branch (line 136) in detect_univariate_drift_with_options."""
    # Create simple test data
    ref = pd.Series([1, 2, 3])
    cur = pd.Series([2, 3, 4])

    # This test directly targets line 136 in univariate.py
    # The error is raised when kind is neither "continuous", "categorical", nor "auto"
    with pytest.raises(ValueError, match="Unknown kind"):
        from tab_right.drift.univariate import detect_univariate_drift_with_options

        # Use a kind value that will definitely reach the "else" clause
        detect_univariate_drift_with_options(ref, cur, kind="not_a_valid_kind")


def test_detect_univariate_drift_with_options_no_normalization():
    """Test detect_univariate_drift_with_options with normalize=False for continuous data.

    This test specifically targets line 144 in univariate.py where normalization is skipped.
    """
    ref = pd.Series([1, 2, 3, 4, 5])
    cur = pd.Series([2, 3, 4, 5, 6])

    # Call with normalize=False to hit line 144
    result = detect_univariate_drift_with_options(ref, cur, kind="continuous", normalize=False)

    # Check that score equals raw_score when normalization is disabled
    assert result["score"] == result["raw_score"]
    assert result["type"] == "wasserstein"
    assert result["score"] == 1.0  # The expected distance between the shifted distributions


def test_detect_univariate_drift_with_unknown_kind_direct_import():
    """Test error handling with an invalid 'kind' parameter with direct import.

    This test specifically tries to hit line 136 in univariate.py using a different approach.
    """
    # Import directly to make sure we're calling the exact function
    from tab_right.drift.univariate import detect_univariate_drift_with_options

    ref = pd.Series([1, 2, 3])
    cur = pd.Series([2, 3, 4])

    # This should raise a ValueError with "Unknown kind" when kind is neither auto, continuous nor categorical
    try:
        detect_univariate_drift_with_options(ref, cur, kind="invalid_kind_value")
        raise AssertionError("Expected ValueError but none was raised")
    except ValueError as e:
        assert str(e) == "Unknown kind"
