"""Compute Cramér’s V statistic for categorical-categorical association."""

import pandas as pd
import scipy.stats


def cramer_v(
    x: pd.Series,
    y: pd.Series,
) -> float:
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
    # Use pandas operations for sqrt
    return phi2**0.5 / min(k - 1, r - 1) ** 0.5 if min(k - 1, r - 1) > 0 else 0.0
