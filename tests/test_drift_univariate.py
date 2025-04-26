import numpy as np
import pandas as pd
import pytest
from tab_right.drift.univariate import cramer_v, psi, detect_univariate_drift

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
