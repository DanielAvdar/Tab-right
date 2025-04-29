from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Union

import pandas as pd


@dataclass
class DriftCalc(Protocol):
    """Class schema for segmentation performance calculations."""

    df1: pd.DataFrame
    df2: pd.DataFrame
    kind: Union[str, Iterable[bool]] = "auto"

    def __call__(self) -> pd.DataFrame:
        """Calculate drift between two DataFrames."""


categorical_drift_calc = Callable[[pd.Series, pd.Series], float]
continuous_drift_calc = Callable[[pd.Series, pd.Series, int], float]
