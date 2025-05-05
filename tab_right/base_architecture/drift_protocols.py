"""Protocol definitions for drift detection and analysis in tab-right.

This module defines protocol classes and type aliases used for implementing
drift detection functionality across different feature types.
"""

from dataclasses import dataclass
from typing import Iterable, Protocol, Union

import pandas as pd


@dataclass
class DriftCalcP(Protocol):
    """Class schema for segmentation performance calculations.

    This class is used to define the interface for drift calculation methods.
    It includes the DataFrames to be compared and the kind of drift to be detected.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame to compare.
    df2 : pd.DataFrame
        The second DataFrame to compare.
    kind : Union[str, Iterable[bool]], default "auto"
        The type of drift to detect. Can be "auto", "categorical", or "continuous".
        If "auto", the type is inferred from the data types of the columns.
        If an iterable of booleans is provided, 0 for categorical and 1 for continuous,
        must match the number of columns in df1 and df2.

    """

    df1: pd.DataFrame
    df2: pd.DataFrame
    kind: Union[str, Iterable[bool]] = "auto"

    def __call__(
        self,
        columns: Union[None, Iterable[str]] = None,
        bins: int = 4,
    ) -> pd.DataFrame:
        """Calculate drift between two DataFrames.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the drift metrics for each column.
            must contain the following columns:
            - "type": categorical or continuous.
            - "score": The calculated drift score.
            - "feature": The name of the feature.

        """

        # categorical_drift_calc = Callable[[pd.Series, pd.Series], float]
        # continuous_drift_calc = Callable[[pd.Series, pd.Series, int], float]

    def get_prob_density(
        self,
        columns: Union[None, Iterable[str]] = None,
        bins: int = 4,
    ) -> pd.DataFrame:
        """Get the probability density function for the features.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the probability density function for each feature.
            must contain the following columns:
            - "feature": The name of the feature.
            - "prob_density": The calculated probability density function.

        """

    @classmethod
    def _categorical_drift_calc(cls, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate Cramér's V for categorical features. normalized."""
        # Cramér's V is a measure of association between two categorical variables.
        # It ranges from 0 to 1, where 0 indicates no association and 1 indicates perfect association.
        # The formula for Cramér's V is:
        # V = sqrt(χ² / (n * min(k - 1, r - 1)))
        # where χ² is the chi-squared statistic, n is the total number of observations,
        # k is the number of categories in the first variable, and r is the number of categories in the second variable.

    @classmethod
    def _continuous_drift_calc(cls, s1: pd.Series, s2: pd.Series, bins: int) -> float:
        """Calculate PSI for continuous features. normalized."""
        # PSI is a measure of how much the distribution of a feature has changed over time.
        # It is calculated by comparing the distribution of the feature in two different datasets.
        # The formula for PSI is:
        # PSI = sum((p1 - p2) * log(p1 / p2))
        # where p1 and p2 are the probabilities of the feature in the two datasets.
