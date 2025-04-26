"""Univariate drift detection utilities for tab-right drift subpackage."""

from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats  # type: ignore


def cramer_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramér’s V statistic for categorical-categorical association.

    Parameters
    ----------
    x : pd.Series
        First categorical variable.
    y : pd.Series
        Second categorical variable.

    Returns
    -------
    float
        Cramér’s V value in [0, 1].

    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0.0


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) for categorical or binned continuous data.

    Parameters
    ----------
    expected : pd.Series
        Reference distribution.
    actual : pd.Series
        Current distribution.
    bins : int, default 10
        Number of bins for continuous data.

    Returns
    -------
    float
        PSI value (>= 0).

    """
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_perc / np.sum(expected_perc)
    actual_perc = actual_perc / np.sum(actual_perc)
    psi_value = np.sum((actual_perc - expected_perc) * np.log((actual_perc + 1e-8) / (expected_perc + 1e-8)))
    return psi_value


def detect_univariate_drift(reference: pd.Series, current: pd.Series, kind: str = "auto") -> Tuple[str, float]:
    """Detect drift between two 1D distributions.

    Parameters
    ----------
    reference : pd.Series
        Reference distribution.
    current : pd.Series
        Current distribution.
    kind : str, default "auto"
        "auto", "categorical", or "continuous". If "auto", infers from dtype.

    Returns
    -------
    tuple
        (metric name, value)

    Raises
    ------
    ValueError
        If kind is not recognized.

    """
    if kind == "auto":
        if pd.api.types.is_numeric_dtype(reference):
            kind = "continuous"
        else:
            kind = "categorical"
    if kind == "continuous":
        return "wasserstein", scipy.stats.wasserstein_distance(reference, current)
    elif kind == "categorical":
        return "cramer_v", cramer_v(reference, current)
    else:
        raise ValueError("Unknown kind")


def detect_univariate_drift_df(reference: pd.DataFrame, current: pd.DataFrame, kind: str = "auto") -> pd.DataFrame:
    """Detect drift for each column in two DataFrames.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference DataFrame.
    current : pd.DataFrame
        Current DataFrame.
    kind : str, default "auto"
        "auto", "categorical", or "continuous". If "auto", infers from dtype.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, metric, value.

    """
    results = []
    common_cols = set(reference.columns) & set(current.columns)
    for col in common_cols:
        metric, value = detect_univariate_drift(reference[col], current[col], kind=kind)
        results.append({"feature": col, "metric": metric, "value": value})
    return pd.DataFrame(results)
