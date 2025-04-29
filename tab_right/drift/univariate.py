"""Univariate drift detection utilities for tab-right drift subpackage."""

from typing import Tuple

import pandas as pd
import scipy.stats

from tab_right.drift.cramer_v import cramer_v


def detect_univariate_drift(
    reference: pd.Series,
    current: pd.Series,
    kind: str = "auto",
) -> Tuple[str, float]:
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
        # Use pandas to_numpy for scipy
        return "wasserstein", scipy.stats.wasserstein_distance(reference.to_numpy(), current.to_numpy())
    elif kind == "categorical":
        return "cramer_v", cramer_v(reference, current)
    else:
        raise ValueError("Unknown kind")


def detect_univariate_drift_df(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    kind: str = "auto",
) -> pd.DataFrame:
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
