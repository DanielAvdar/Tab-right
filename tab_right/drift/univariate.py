import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, chi2_contingency
from typing import Tuple

def cramer_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér’s V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1)) if min(k-1, r-1) > 0 else 0.0

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for categorical or binned continuous data."""
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_perc / np.sum(expected_perc)
    actual_perc = actual_perc / np.sum(actual_perc)
    psi_value = np.sum((actual_perc - expected_perc) * np.log((actual_perc + 1e-8) / (expected_perc + 1e-8)))
    return psi_value

def detect_univariate_drift(
    reference: pd.Series,
    current: pd.Series,
    kind: str = "auto"
) -> Tuple[str, float]:
    """
    Detect drift between two 1D distributions.
    kind: "auto", "categorical", "continuous"
    Returns: metric name, value
    """
    if kind == "auto":
        if pd.api.types.is_numeric_dtype(reference):
            kind = "continuous"
        else:
            kind = "categorical"
    if kind == "continuous":
        return "wasserstein", wasserstein_distance(reference, current)
    elif kind == "categorical":
        return "cramer_v", cramer_v(reference, current)
    else:
        raise ValueError("Unknown kind")
