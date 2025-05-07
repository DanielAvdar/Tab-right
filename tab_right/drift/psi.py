"""some doc here."""

import numpy as np
import pandas as pd


def psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
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
    # Use pandas cut and value_counts for binning and proportions
    expected_bins = pd.cut(expected, bins=bins, duplicates="drop")
    actual_bins = pd.cut(actual, bins=bins, duplicates="drop")
    expected_perc = expected_bins.value_counts(sort=False, normalize=True)
    actual_perc = actual_bins.value_counts(sort=False, normalize=True)
    # Align indexes to ensure same bins
    expected_perc, actual_perc = expected_perc.align(actual_perc, fill_value=1e-8)
    psi_value = ((actual_perc - expected_perc) * ((actual_perc + 1e-8) / (expected_perc + 1e-8)).apply(np.log)).sum()
    return psi_value
