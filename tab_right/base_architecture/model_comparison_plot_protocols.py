"""Protocols for model comparison visualization in tab-right.

This module defines protocols for visualizing model comparison results. It provides
interfaces for creating visualizations that compare multiple prediction datasets
against true labels.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Tuple, Union, runtime_checkable

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from .model_comparison_protocols import PredictionCalculationP

Figure = Union[go.Figure, plt.Figure]


@runtime_checkable
@dataclass
class ModelComparisonPlotP(Protocol):
    """Protocol for model comparison visualization implementations.

    This protocol defines the interface for visualizing comparison results
    between multiple predictions and true labels.

    Parameters
    ----------
    comparison_calc : PredictionCalculationP
        An implementation of PredictionCalculationP that provides the comparison
        metrics to visualize.

    """

    comparison_calc: PredictionCalculationP

    def plot_error_distribution(
        self,
        pred_data: List[pd.Series],
        model_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        bins: int = 30,
        **kwargs: Any,
    ) -> Figure:
        """Create a visualization comparing error distributions across models.

        Parameters
        ----------
        pred_data : List[pd.Series]
            List of prediction Series to compare against the label.
        model_names : Optional[List[str]], default None
            Names for each model. If None, uses default names like "Model 0", "Model 1", etc.
        figsize : Tuple[int, int], default (12, 8)
            Figure size as (width, height) in inches.
        bins : int, default 30
            Number of bins for histogram visualization.
        **kwargs : Any
            Additional parameters for the plotting implementation.

        Returns
        -------
        Figure
            A figure object containing the error distribution comparison.

        """
        ...

    def plot_pairwise_comparison(
        self,
        pred_data: List[pd.Series],
        model_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        **kwargs: Any,
    ) -> Figure:
        """Create a pairwise comparison plot between models.

        Parameters
        ----------
        pred_data : List[pd.Series]
            List of prediction Series to compare against the label.
        model_names : Optional[List[str]], default None
            Names for each model. If None, uses default names.
        figsize : Tuple[int, int], default (10, 8)
            Figure size as (width, height) in inches.
        **kwargs : Any
            Additional parameters for the plotting implementation.

        Returns
        -------
        Figure
            A figure object containing the pairwise comparison visualization.

        """
        ...

    def plot_model_performance_summary(
        self,
        pred_data: List[pd.Series],
        model_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        **kwargs: Any,
    ) -> Figure:
        """Create a summary visualization of model performance metrics.

        Parameters
        ----------
        pred_data : List[pd.Series]
            List of prediction Series to compare against the label.
        model_names : Optional[List[str]], default None
            Names for each model. If None, uses default names.
        metrics : Optional[List[str]], default None
            List of metrics to display. If None, uses default metrics.
        figsize : Tuple[int, int], default (12, 6)
            Figure size as (width, height) in inches.
        **kwargs : Any
            Additional parameters for the plotting implementation.

        Returns
        -------
        Figure
            A figure object containing the performance summary visualization.

        """
        ...
