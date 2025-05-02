"""Protocol definitions for drift detection and analysis in tab-right.

This module defines protocol classes and type aliases used for implementing
drift detection functionality across different feature types.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Union

import pandas as pd


@dataclass
class DriftCalc(Protocol):
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

    def __call__(self) -> pd.DataFrame:
        """Calculate drift between two DataFrames.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the drift metrics for each column.
            must contain the following columns:
            - "type": psi or cramer_v
            - "score": The calculated drift score.
            - "feature": The name of the feature.

        """


categorical_drift_calc = Callable[[pd.Series, pd.Series], float]
continuous_drift_calc = Callable[[pd.Series, pd.Series, int], float]
