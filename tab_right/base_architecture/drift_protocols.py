"""Protocol definitions for drift detection and analysis in tab-right.

This module defines protocol classes and type aliases used for implementing
drift detection functionality across different feature types. These protocols
establish a consistent interface for all drift detection implementations.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Protocol, Union, runtime_checkable

import pandas as pd


@runtime_checkable
@dataclass
class DriftCalcP(Protocol):
    """Protocol for drift calculation implementations.

    This protocol defines the interface that all drift calculation classes must implement.
    It specifies methods for detecting distributional shifts between two datasets.

    Parameters
    ----------
    df1 : pd.DataFrame
        The reference DataFrame containing the baseline distribution.
    df2 : pd.DataFrame
        The current DataFrame to compare against the reference.
    kind : Union[str, Iterable[bool], Dict[str, str]], default "auto"
        Controls how columns are treated:
        - "auto": Automatically infer from data types (numeric as continuous, others as categorical)
        - "categorical": Treat all columns as categorical
        - "continuous": Treat all columns as continuous
        - Iterable[bool]: Specification for each column (True for continuous, False for categorical)
        - Dict[str, str]: Explicit mapping from column name to "continuous" or "categorical"

    Notes
    -----
    Implementations of this protocol are responsible for:
    1. Comparing distribution shifts between reference and current data
    2. Automatically selecting appropriate metrics based on data types
    3. Providing normalized scores for comparison across features

    """

    df1: pd.DataFrame
    df2: pd.DataFrame
    kind: Union[str, Iterable[bool], Dict[str, str]] = "auto"

    def __call__(self, columns: Optional[Iterable[str]] = None, bins: int = 10, **kwargs: Mapping) -> pd.DataFrame:
        """Calculate drift between two DataFrames.

        Parameters
        ----------
        columns : Optional[Iterable[str]], default None
            Specific columns to analyze. If None, analyzes all common columns.
        bins : int, default 10
            Number of bins for histograms when analyzing continuous features.
        **kwargs : Mapping
            Additional parameters specific to the drift calculation implementation.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the drift metrics for each column.
            Must contain at least the following columns:
            - "feature": The name of the feature.
            - "type": The type of metric used (e.g., "wasserstein", "cramer_v", "psi").
            - "score": The calculated drift score (normalized to [0,1] when applicable).

            May also include:
            - "raw_score": The original, unnormalized score for continuous features.
            - "threshold": Optional threshold value for drift significance.

        """

    def get_prob_density(
        self,
        columns: Optional[Iterable[str]] = None,
        bins: int = 10,
    ) -> pd.DataFrame:
        """Get the probability density functions for the features.

        Parameters
        ----------
        columns : Optional[Iterable[str]], default None
            Specific columns to analyze. If None, analyzes all common columns.
        bins : int, default 10
            Number of bins for histograms when analyzing continuous features.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the probability density functions.
            Must contain at least the following columns:
            - "feature": The name of the feature.
            - "bin": The bin or category.
            - "ref_density": The density in the reference dataset.
            - "cur_density": The density in the current dataset.

        """

    @classmethod
    def _categorical_drift_calc(cls, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate drift for categorical features (normalized to [0,1]).

        Parameters
        ----------
        s1 : pd.Series
            Reference distribution.
        s2 : pd.Series
            Current distribution.

        Returns
        -------
        float
            Normalized drift score between 0 and 1.

        Notes
        -----
        Typically implemented using Cramér's V statistic:
        V = sqrt(χ² / (n * min(k-1, r-1)))
        where:
        - χ² is the chi-squared statistic
        - n is the total number of observations
        - k is the number of categories in the first variable
        - r is the number of categories in the second variable

        """

    @classmethod
    def _continuous_drift_calc(cls, s1: pd.Series, s2: pd.Series, bins: int = 10) -> float:
        """Calculate drift for continuous features (normalized to [0,1]).

        Parameters
        ----------
        s1 : pd.Series
            Reference distribution.
        s2 : pd.Series
            Current distribution.
        bins : int, default 10
            Number of bins for histogram comparison.

        Returns
        -------
        float
            Normalized drift score between 0 and 1.

        Notes
        -----
        Can be implemented using various metrics:
        - Wasserstein distance (normalized)
        - Population Stability Index (PSI)
        - Kolmogorov-Smirnov test statistic

        """
