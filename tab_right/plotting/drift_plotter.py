"""Implementation of the DriftPlotP protocol using Matplotlib."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tab_right.base_architecture.drift_protocols import DriftCalcP

from ._matplotlib_backend import (
    plot_categorical_feature,
    plot_continuous_feature,
    plot_drift_values as plot_drift_values_mp,
    plot_feature_drift_kde_mp,
    plot_multiple_features,
)
from ._plotly_backend import plot_drift_values, plot_feature_drift_kde
from ._plotting_utils import create_empty_figure, validate_column_exists


@dataclass
class DriftPlotter:
    """Implementation of DriftPlotP protocol using Matplotlib."""

    drift_calc: DriftCalcP

    def __post_init__(self) -> None:
        """Initialize the DriftPlotter with validation.

        Raises:
            TypeError: If drift_calc is not an instance of DriftCalcP.
            ValueError: If either dataframe is empty.

        """
        if not isinstance(self.drift_calc, DriftCalcP):
            raise TypeError("drift_calc must be an instance of DriftCalcP")
        if self.drift_calc.df1.empty or self.drift_calc.df2.empty:
            raise ValueError("Both dataframes must be non-empty.")
        # Add a _feature_types attribute that can be used by mypy
        self._feature_types = getattr(self.drift_calc, "_feature_types", {})

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
    ) -> plt.Figure:
        """Create a bar chart visualization of drift across multiple features.

        Args:
            columns: Specific columns to plot drift for. If None, all available columns are used.
            bins: Number of bins to use for continuous features.
            figsize: Figure size as (width, height) in inches.
            sort_by: Column to sort results by (usually "score").
            ascending: Whether to sort in ascending order.
            top_n: If specified, only show the top N features.
            threshold: If specified, mark features above this threshold in a different color.
            **kwargs: Additional arguments passed to the drift calculator.

        Returns:
            A matplotlib Figure object containing the generated plot.

        """
        drift_results = self.drift_calc(columns=columns, bins=bins, **kwargs)
        return plot_multiple_features(
            drift_results=drift_results,
            figsize=figsize,
            sort_by=sort_by,
            ascending=ascending,
            top_n=top_n,
            threshold=threshold,
        )

    def plot_single(
        self,
        column: str,
        bins: int = 10,
        figsize: Tuple[int, int] = (10, 6),
        show_metrics: bool = True,
        **kwargs: Any,
    ) -> plt.Figure:
        """Create a detailed visualization of drift for a single feature.

        Args:
            column: The column/feature to visualize.
            bins: Number of bins to use for continuous features.
            figsize: Figure size as (width, height) in inches.
            show_metrics: Whether to show drift metrics in the plot.
            **kwargs: Additional arguments passed to the drift calculator.

        Returns:
            A matplotlib Figure object containing the generated plot.

        """
        validate_column_exists(column, self._feature_types)

        col_type = self._feature_types[column]
        density_df = self.drift_calc.get_prob_density(columns=[column], bins=bins)
        drift_metrics = self.drift_calc(columns=[column], bins=bins)

        if density_df.empty:
            return create_empty_figure(figsize=figsize, message=f"No data available for column '{column}'.")

        feature_density = density_df[density_df["feature"] == column]
        bins_or_cats = np.asarray(feature_density["bin"].values)
        ref_density = np.asarray(feature_density["ref_density"].values)
        cur_density = np.asarray(feature_density["cur_density"].values)

        if col_type == "categorical":
            return plot_categorical_feature(
                feature_density=feature_density,
                bins_or_cats=bins_or_cats,
                ref_density=ref_density,
                cur_density=cur_density,
                column=column,
                figsize=figsize,
                show_metrics=show_metrics,
                drift_metrics=drift_metrics,
            )
        else:  # continuous
            return plot_continuous_feature(
                feature_density=feature_density,
                bins_or_cats=bins_or_cats,
                ref_density=ref_density,
                cur_density=cur_density,
                column=column,
                figsize=figsize,
                show_metrics=show_metrics,
                drift_metrics=drift_metrics,
            )

    def get_distribution_plots(
        self, columns: Optional[Iterable[str]] = None, bins: int = 10, **kwargs: Any
    ) -> Dict[str, plt.Figure]:
        """Generate individual distribution comparison plots for multiple features.

        Args:
            columns: Specific columns to generate plots for. If None, all available columns are used.
            bins: Number of bins to use for continuous features.
            **kwargs: Additional arguments passed to plot_single.

        Returns:
            A dictionary mapping column names to their respective matplotlib Figure objects.

        """
        if columns is None:
            columns = list(self._feature_types.keys())
        else:
            columns = [col for col in columns if col in self._feature_types]

        plots = {}
        for col in columns:
            try:
                # Create plot but don't show it immediately
                fig = self.plot_single(column=col, bins=bins, show_metrics=True, **kwargs)
                plots[col] = fig
            except Exception as e:
                print(f"Could not generate plot for column '{col}': {e}")
                # Optionally create a placeholder figure indicating error
                plots[col] = create_empty_figure(figsize=(10, 6), message=f"Error plotting {col}")

        # Store the figures but don't close them yet - they're still needed for return
        result = {k: fig for k, fig in plots.items()}

        # Close all figures including any that might have been created internally
        plt.close("all")

        return result

    def plot_drift(
        self,
        drift_df: pd.DataFrame,
        value_col: str = "value",
        feature_col: str = "feature",
    ) -> go.Figure:
        """Plot drift values for each feature as a bar chart using Plotly.

        Args:
            drift_df: DataFrame with drift results. Should contain columns for feature names and drift values.
            value_col: Name of the column containing drift values.
            feature_col: Name of the column containing feature names.

        Returns:
            go.Figure: Plotly bar chart of drift values by feature.

        """
        return plot_drift_values(
            drift_df=drift_df,
            value_col=value_col,
            feature_col=feature_col,
        )

    def plot_drift_mp(
        self,
        drift_df: pd.DataFrame,
        value_col: str = "value",
        feature_col: str = "feature",
    ) -> plt.Figure:
        """Plot drift values for each feature as a bar chart using Matplotlib.

        Args:
            drift_df: DataFrame with drift results. Should contain columns for feature names and drift values.
            value_col: Name of the column containing drift values.
            feature_col: Name of the column containing feature names.

        Returns:
            plt.Figure: Matplotlib figure with bar chart of drift values by feature.

        """
        return plot_drift_values_mp(
            drift_df=drift_df,
            value_col=value_col,
            feature_col=feature_col,
        )

    def plot_feature_drift_kde(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str = None,
        show_score: bool = True,
        ref_label: str = "Train Dataset",
        cur_label: str = "Test Dataset",
        normalize: bool = True,
        normalization_method: str = "range",
        show_raw_score: bool = False,
    ) -> go.Figure:
        """Plot distribution drift for a single feature using KDE with Plotly.

        Args:
            reference: Reference (train) data for the feature.
            current: Current (test) data for the feature.
            feature_name: Name of the feature (for labeling plots).
            show_score: Whether to display the drift score annotation.
            ref_label: Label for the reference data.
            cur_label: Label for the current data.
            normalize: Whether to normalize the Wasserstein distance.
            normalization_method: Method to use for normalization: "range", "std", or "iqr".
            show_raw_score: Whether to show both normalized and raw scores.

        Returns:
            go.Figure: Plotly figure with overlaid histograms, means, medians, and drift score annotation.

        """
        return plot_feature_drift_kde(
            reference=reference,
            current=current,
            feature_name=feature_name,
            show_score=show_score,
            ref_label=ref_label,
            cur_label=cur_label,
            normalize=normalize,
            normalization_method=normalization_method,
            show_raw_score=show_raw_score,
        )

    def plot_feature_drift_kde_mp(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str = None,
        show_score: bool = True,
        ref_label: str = "Train Dataset",
        cur_label: str = "Test Dataset",
        normalize: bool = True,
        normalization_method: str = "range",
        show_raw_score: bool = False,
    ) -> plt.Figure:
        """Plot distribution drift for a single feature using KDE with Matplotlib.

        Args:
            reference: Reference (train) data for the feature.
            current: Current (test) data for the feature.
            feature_name: Name of the feature (for labeling plots).
            show_score: Whether to display the drift score annotation.
            ref_label: Label for the reference data.
            cur_label: Label for the current data.
            normalize: Whether to normalize the Wasserstein distance.
            normalization_method: Method to use for normalization: "range", "std", or "iqr".
            show_raw_score: Whether to show both normalized and raw scores.

        Returns:
            plt.Figure: Matplotlib figure with overlaid distributions, means, and drift score.

        """
        return plot_feature_drift_kde_mp(
            reference=reference,
            current=current,
            feature_name=feature_name,
            show_score=show_score,
            ref_label=ref_label,
            cur_label=cur_label,
            normalize=normalize,
            normalization_method=normalization_method,
            show_raw_score=show_raw_score,
        )
