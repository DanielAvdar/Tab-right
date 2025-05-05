"""Protocols for drift visualization in tab-right.

This module defines protocols for visualizing drift between datasets. It provides
interfaces for creating both single-feature and multi-feature drift visualizations.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, Tuple, Union, runtime_checkable

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .drift_protocols import DriftCalcP


@runtime_checkable
@dataclass
class DriftPlotP(Protocol):
    """Protocol for drift visualization implementations.

    This protocol defines the interface that all drift visualization classes must implement.
    It specifies methods for creating visualizations of distribution shifts between datasets.

    Parameters
    ----------
    drift_calc : DriftCalcP
        An implementation of DriftCalcP that provides the drift metrics to visualize.

    """

    drift_calc: DriftCalcP

    def plot_multiple(
        self,
        columns: Optional[Iterable[str]] = None,
        bins: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        sort_by: str = "score",
        ascending: bool = False,
        top_n: Optional[int] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[go.Figure, plt.Figure]:
        """Create a visualization of drift across multiple features.

        Parameters
        ----------
        columns : Optional[Iterable[str]], default None
            Specific columns to visualize. If None, visualizes all common columns.
        bins : int, default 10
            Number of bins for histograms when visualizing continuous features.
        figsize : Tuple[int, int], default (12, 8)
            Figure size as (width, height) in inches.
        sort_by : str, default "score"
            Column to sort the results by, typically "score" or "feature".
        ascending : bool, default False
            Whether to sort in ascending or descending order.
        top_n : Optional[int], default None
            If provided, only shows the top N features with highest drift.
        threshold : Optional[float], default None
            If provided, highlights features with drift above this threshold.
        **kwargs : Any
            Additional parameters for the plotting implementation.

        Returns
        -------
        Union[go.Figure, plt.Figure]
            A figure object containing the drift visualization.

        """
        ...

    def plot_single(
        self, column: str, bins: int = 10, figsize: Tuple[int, int] = (10, 6), show_metrics: bool = True, **kwargs: Any
    ) -> Union[go.Figure, plt.Figure]:
        """Create a detailed visualization of drift for a single feature.

        Parameters
        ----------
        column : str
            The specific column to visualize.
        bins : int, default 10
            Number of bins for histograms when visualizing continuous features.
        figsize : Tuple[int, int], default (10, 6)
            Figure size as (width, height) in inches.
        show_metrics : bool, default True
            Whether to display drift metrics on the plot.
        **kwargs : Any
            Additional parameters for the plotting implementation.

        Returns
        -------
        Union[go.Figure, plt.Figure]
            A figure object containing the drift visualization.

        """
        ...

    def get_distribution_plots(
        self, columns: Optional[Iterable[str]] = None, bins: int = 10, **kwargs: Any
    ) -> Dict[str, Union[go.Figure, plt.Figure]]:
        """Generate individual distribution comparison plots for multiple features.

        Parameters
        ----------
        columns : Optional[Iterable[str]], default None
            Specific columns to visualize. If None, visualizes all common columns.
        bins : int, default 10
            Number of bins for histograms when visualizing continuous features.
        **kwargs : Any
            Additional parameters for the plotting implementation.

        Returns
        -------
        Dict[str, Union[go.Figure, plt.Figure]]
            A dictionary mapping feature names to their distribution comparison plots.

        """
        ...
