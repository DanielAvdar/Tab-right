import numpy as np
import pandas as pd
from tab_right.drift.univariate import cramer_v, psi, detect_univariate_drift

def test_cramer_v_basic():
    x = pd.Series(["a", "b", "a", "b", "c", "c"])
    y = pd.Series(["a", "a", "b", "b", "c", "c"])
    v = cramer_v(x, y)
    assert 0 <= v <= 1

def test_psi_basic():
    expected = pd.Series([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    actual = pd.Series([0, 1, 1, 2, 2, 2, 3, 3, 3, 0])
    value = psi(expected, actual, bins=4)
    assert value >= 0

def test_detect_univariate_drift_continuous():
    ref = pd.Series(np.random.normal(0, 1, 100))
    cur = pd.Series(np.random.normal(0.5, 1, 100))
    metric, value = detect_univariate_drift(ref, cur, kind="continuous")
    assert metric == "wasserstein"
    assert value >= 0

def test_detect_univariate_drift_categorical():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 30 + ["b"] * 70)
    metric, value = detect_univariate_drift(ref, cur, kind="categorical")
    assert metric == "cramer_v"
    assert 0 <= value <= 1

def test_detect_univariate_drift_auto():
    ref = pd.Series([1, 2, 3, 4, 5])
    cur = pd.Series([1, 2, 2, 4, 5])
    metric, value = detect_univariate_drift(ref, cur)
    assert metric in ("wasserstein", "cramer_v")
