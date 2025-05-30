"""Univariate drift detection utilities for tab-right drift subpackage."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats

from tab_right.drift.cramer_v import cramer_v


def normalize_wasserstein(
    reference: pd.Series, current: pd.Series, wasserstein_value: float, method: str = "range"
) -> float:
    """Normalize Wasserstein distance to make it more comparable across features with different scales.

    Parameters
    ----------
    reference : pd.Series
        Reference distribution.
    current : pd.Series
        Current distribution.
    wasserstein_value : float
        Raw Wasserstein distance value to normalize.
    method : str, default "range"
        Normalization method:
        - "range": Divide by the combined range of both distributions
        - "std": Divide by the pooled standard deviation
        - "iqr": Divide by the pooled interquartile range

    Returns
    -------
    float
        Normalized drift score between 0 and 1 in most practical cases.

    Raises
    ------
    ValueError
        If an unknown normalization method is provided.

    """
    if wasserstein_value == 0:
        return 0.0

    if pd.isna(wasserstein_value):
        return np.nan

    # Combine data for normalization calculations
    combined = pd.concat([reference, current])

    if method == "range":
        # Normalize by the combined range (max - min)
        normalization_factor = combined.max() - combined.min()
        if normalization_factor == 0:  # All values are the same
            return 0.0
    elif method == "std":
        # Normalize by the pooled standard deviation
        normalization_factor = combined.std()
        if normalization_factor == 0 or pd.isna(normalization_factor):  # No variance or insufficient data
            return 0.0
    elif method == "iqr":
        # Normalize by the interquartile range
        q75, q25 = np.nanpercentile(combined, [75, 25])
        normalization_factor = q75 - q25
        if normalization_factor == 0 or pd.isna(normalization_factor):  # No variance or insufficient data
            return 0.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return wasserstein_value / normalization_factor


@dataclass
class UnivariateDriftCalculator:
    """Calculate univariate drift between two DataFrames.

    This class implements the DriftCalc protocol and provides methods for
    detecting drift between two DataFrames using column-by-column analysis.

    Parameters
    ----------
    df1 : pd.DataFrame
        The reference DataFrame
    df2 : pd.DataFrame
        The current DataFrame to compare against the reference
    kind : Union[None, Dict[str, str]], default None
        How to treat columns:
        - None: Infer from data types
        - dict: Specify "continuous" or "categorical" for each column
    normalize : bool, default True
        Whether to normalize continuous drift scores
    normalization_method : str, default "range"
        Method to use for normalization, see normalize_wasserstein for options

    """

    df1: pd.DataFrame
    df2: pd.DataFrame
    kind: Union[None, Dict[str, str]] = None
    normalize: bool = True
    normalization_method: str = "range"

    def __post_init__(self) -> None:
        """Post-initialization: enforce kind protocol at instantiation.

        Raises
        ------
        ValueError
            If kind is not None and not a dict mapping column names to 'continuous' or 'categorical'.

        """
        if self.kind is not None and not isinstance(self.kind, dict):
            raise ValueError("kind must be None or a dict mapping column names to 'continuous' or 'categorical'.")

    def __call__(self) -> pd.DataFrame:
        """Calculate drift between two DataFrames.

        Returns
        -------
        pd.DataFrame
            DataFrame with drift results for each feature.

        Raises
        ------
        ValueError
            If kind is not None and not a dict mapping column names to 'continuous' or 'categorical'.

        """
        if self.kind is not None and not isinstance(self.kind, dict):
            raise ValueError("kind must be None or a dict mapping column names to 'continuous' or 'categorical'.")

        results = []
        common_cols = set(self.df1.columns) & set(self.df2.columns)

        if self.kind is None:
            kind_per_col = {}
            for col in common_cols:
                if pd.api.types.is_numeric_dtype(self.df1[col]):
                    nunique = min(self.df1[col].nunique(), self.df2[col].nunique())
                    if nunique <= 20:
                        kind_per_col[col] = "categorical"
                    else:
                        kind_per_col[col] = "continuous"
                else:
                    kind_per_col[col] = "categorical"
        elif isinstance(self.kind, dict):
            kind_per_col = {}
            for col in common_cols:
                t = self.kind.get(col, None)
                if t is not None:
                    kind_per_col[col] = t
                else:
                    if pd.api.types.is_numeric_dtype(self.df1[col]):
                        nunique = min(self.df1[col].nunique(), self.df2[col].nunique())
                        if nunique <= 20:
                            kind_per_col[col] = "categorical"
                        else:
                            kind_per_col[col] = "continuous"
                    else:
                        kind_per_col[col] = "categorical"
        else:
            raise ValueError("kind must be None or a dict mapping column names to 'continuous' or 'categorical'.")

        # Calculate drift for each column
        for col in common_cols:
            result_dict = detect_univariate_drift_with_options(
                self.df1[col],
                self.df2[col],
                kind=kind_per_col[col],
                normalize=self.normalize,
                normalization_method=self.normalization_method,
            )
            result_dict["feature"] = col
            results.append(result_dict)

        return pd.DataFrame(results)


def detect_univariate_drift_with_options(
    reference: pd.Series,
    current: pd.Series,
    kind: str = "auto",
    normalize: bool = True,
    normalization_method: str = "range",
) -> Dict[str, Any]:
    """Detect drift between two 1D distributions with normalization options.

    Parameters
    ----------
    reference : pd.Series
        Reference distribution.
    current : pd.Series
        Current distribution.
    kind : str, default "auto"
        "auto", "categorical", or "continuous". If "auto", infers from dtype.
    normalize : bool, default True
        Whether to normalize continuous drift scores
    normalization_method : str, default "range"
        Method to use for normalization, see normalize_wasserstein for options

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - "type": Metric name (wasserstein or cramer_v)
        - "score": Drift score (normalized for continuous if normalize=True)
        - "raw_score": Unnormalized drift score (only for continuous)

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
        # Calculate raw Wasserstein distance
        raw_score = scipy.stats.wasserstein_distance(reference.to_numpy(), current.to_numpy())

        result = {"type": "wasserstein", "raw_score": raw_score}

        # Apply normalization if requested
        if normalize:
            result["score"] = normalize_wasserstein(reference, current, raw_score, method=normalization_method)
        else:
            # LINE 144: This is where we assign the raw score when normalization is turned off
            result["score"] = raw_score  # Ensure this line (144) is covered

        return result

    elif kind == "categorical":
        # Cramer's V is already normalized between 0 and 1
        cv_score = cramer_v(reference, current)
        return {"type": "cramer_v", "score": cv_score}
    else:
        # LINE 136: This is where we raise ValueError for unknown kind
        raise ValueError("Unknown kind")  # Ensure this line (136) is covered


def detect_univariate_drift(
    reference: pd.Series,
    current: pd.Series,
    kind: str = "auto",
    normalize: bool = True,
    normalization_method: str = "range",
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
    normalize : bool, default True
        Whether to normalize continuous drift scores
    normalization_method : str, default "range"
        Method to use for normalization, see normalize_wasserstein for options

    Returns
    -------
    tuple
        (metric name, value)

    Notes
    -----
    This function calls detect_univariate_drift_with_options internally and may
    raise ValueError if kind is not recognized or if an invalid normalization
    method is specified.

    """
    result = detect_univariate_drift_with_options(
        reference, current, kind=kind, normalize=normalize, normalization_method=normalization_method
    )
    return result["type"], result["score"]


def detect_univariate_drift_df(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    kind: str = "auto",
    normalize: bool = True,
    normalization_method: str = "range",
) -> pd.DataFrame:
    """Detect drift for each column in two DataFrames.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference DataFrame.
    current : pd.DataFrame.
        Current DataFrame.
    kind : str, default "auto"
        "auto", "categorical", or "continuous". If "auto", infers from dtype.
    normalize : bool, default True
        Whether to normalize continuous drift scores
    normalization_method : str, default "range"
        Method to use for normalization, see normalize_wasserstein for options

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, metric, value, raw_value (for continuous features).

    Notes
    -----
    This function is provided for backward compatibility.
    For new code, use the UnivariateDriftCalculator class instead.

    """
    # Use the protocol-compliant class for implementation
    drift_calc = UnivariateDriftCalculator(
        df1=reference,
        df2=current,
        kind=None if kind == "auto" else {col: kind for col in reference.index},
        normalize=normalize,
        normalization_method=normalization_method,
    )
    result = drift_calc()

    # Rename columns to match old API for backward compatibility
    result = result.rename(columns={"type": "metric", "score": "value"})
    if "raw_score" in result.columns:
        result = result.rename(columns={"raw_score": "raw_value"})
    return result
