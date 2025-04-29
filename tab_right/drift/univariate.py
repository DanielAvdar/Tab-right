"""Univariate drift detection utilities for tab-right drift subpackage."""

from dataclasses import dataclass
from typing import Iterable, Tuple, Union

import pandas as pd
import scipy.stats

from tab_right.base.drifts_protocols import DriftCalc
from tab_right.drift.cramer_v import cramer_v


@dataclass
class UnivariateDriftCalculator(DriftCalc):
    """Calculate univariate drift between two DataFrames.

    This class implements the DriftCalc protocol and provides methods for
    detecting drift between two DataFrames using column-by-column analysis.

    Parameters
    ----------
    df1 : pd.DataFrame
        The reference DataFrame
    df2 : pd.DataFrame
        The current DataFrame to compare against the reference
    kind : Union[str, Iterable[bool]], default "auto"
        How to treat columns:
        - "auto": Infer from data types
        - "categorical": Treat all columns as categorical
        - "continuous": Treat all columns as continuous
        - Iterable[bool]: Specification for each column (True for continuous, False for categorical)

    """

    df1: pd.DataFrame
    df2: pd.DataFrame
    kind: Union[str, Iterable[bool]] = "auto"

    def __call__(self) -> pd.DataFrame:
        """Calculate drift between two DataFrames.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the drift metrics for each column with columns:
            - "feature": The name of the feature
            - "type": The type of metric used (wasserstein or cramer_v)
            - "score": The calculated drift score

        Raises
        ------
        ValueError
            If the length of kind parameter doesn't match the number of common columns.

        """
        results = []
        common_cols = set(self.df1.columns) & set(self.df2.columns)

        # Convert kind to per-column specification if it's a string
        if isinstance(self.kind, str):
            kind_per_col = {}
            for col in common_cols:
                if self.kind == "auto":
                    # Infer from data type
                    kind_per_col[col] = "continuous" if pd.api.types.is_numeric_dtype(self.df1[col]) else "categorical"
                else:
                    kind_per_col[col] = self.kind
        else:
            # If kind is an iterable of booleans, map to column names
            kind_list = list(self.kind)  # Convert to list for len() operation
            if len(kind_list) != len(common_cols):
                raise ValueError(
                    f"Length of kind ({len(kind_list)}) must match number of common columns ({len(common_cols)})"
                )
            kind_per_col = dict(zip(common_cols, ["continuous" if k else "categorical" for k in kind_list]))

        # Calculate drift for each column
        for col in common_cols:
            metric, value = detect_univariate_drift(self.df1[col], self.df2[col], kind=kind_per_col[col])
            results.append({"feature": col, "type": metric, "score": value})

        return pd.DataFrame(results)


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

    Notes
    -----
    This function is provided for backward compatibility.
    For new code, use the UnivariateDriftCalculator class instead.

    """
    # Use the protocol-compliant class for implementation
    drift_calc = UnivariateDriftCalculator(df1=reference, df2=current, kind=kind)
    result = drift_calc()

    # Rename columns to match old API for backward compatibility
    result = result.rename(columns={"type": "metric", "score": "value"})
    return result
