import numpy as np
import pandas as pd
import pytest

from tab_right.drift.cramer_v import cramer_v
from tab_right.drift.psi import psi
from tab_right.drift.univariate import detect_univariate_drift


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
    metric, value = detect_univariate_drift(ref, cur)
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
        # Shifted continuous
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
        metric, value = detect_univariate_drift(ref, cur, kind=kind)
        assert metric == expected_metric
        if expect_nan:
            assert pd.isna(value)
        else:
            assert expected_range[0] <= value <= expected_range[1]


def test_detect_univariate_drift_df_variety():
    df_ref = pd.DataFrame({"num": [1, 2, 3, 4, 5], "cat": ["a", "b", "a", "b", "c"]})
    df_cur = pd.DataFrame({"num": [2, 3, 4, 5, 6], "cat": ["a", "b", "b", "b", "c"]})
    from tab_right.drift.univariate import detect_univariate_drift_df

    result = detect_univariate_drift_df(df_ref, df_cur)
    assert set(result["feature"]) == {"num", "cat"}
    assert all(m in ("wasserstein", "cramer_v") for m in result["metric"])
    assert all((v >= 0 or pd.isna(v)) for v in result["value"])
