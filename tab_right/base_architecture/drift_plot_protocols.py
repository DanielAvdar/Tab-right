"""TODO UPDATE."""

from dataclasses import dataclass
from typing import Iterable, Protocol, Union

import pandas as pd

from .drift_protocols import DriftCalcP


@dataclass
class DriftPlotP(Protocol):
    """Class schema for drift plotting."""

    drift_calc: DriftCalcP

    def plot_multiple(
        self,
        columns: Union[None, Iterable[str]] = None,
        bins: int = 4,
    ) -> pd.DataFrame:
        """Plot the drift for multiple features."""

    def plot_single(
        self,
        column: str,
        bins: int = 4,
    ) -> pd.DataFrame:
        """Plot the drift for a single feature."""
