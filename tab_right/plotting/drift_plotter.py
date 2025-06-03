"""Implementation of the DriftPlotP protocol using Matplotlib."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from tab_right.base_architecture.drift_protocols import DriftCalcP

from ._matplotlib_backend import (
    plot_categorical_feature,
    plot_continuous_feature,
    plot_drift_values as plot_drift_values_mp,
    plot_multiple_features,
)
from ._plotly_backend import plot_drift_values
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

    @staticmethod
    def plot_feature_drift(
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
        """Plot distribution drift for a single feature using Plotly.

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
            go.Figure: Plotly figure with overlaid distributions, means, and drift score.

        """
        feature_name = feature_name or str(reference.name) if reference.name is not None else "feature"
        drift_score = None
        raw_score = None

        if len(reference) > 0 and len(current) > 0:
            # Import here to avoid circular imports
            from tab_right.drift.univariate import detect_univariate_drift_with_options

            # Get both raw and normalized scores
            result = detect_univariate_drift_with_options(
                reference, current, kind="continuous", normalize=normalize, normalization_method=normalization_method
            )

            drift_score = result["score"]
            if "raw_score" in result:
                raw_score = result["raw_score"]

        # Compute KDEs for smooth lines
        x_min = min(reference.min() if len(reference) else 0, current.min() if len(current) else 0)
        x_max = max(reference.max() if len(reference) else 1, current.max() if len(current) else 1)
        x_grid = np.linspace(x_min, x_max, 200)
        fig = go.Figure()
        if len(reference) > 1:
            kde_ref = gaussian_kde(reference)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=kde_ref(x_grid),
                    mode="lines",
                    name=ref_label,
                    line=dict(color="blue"),
                )
            )
        if len(current) > 1:
            kde_cur = gaussian_kde(current)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=kde_cur(x_grid),
                    mode="lines",
                    name=cur_label,
                    line=dict(color="orange"),
                )
            )
        # Means only (remove medians)
        for arr, color, label, dash in [
            (reference, "blue", f"{ref_label} Mean", "dash"),
            (current, "orange", f"{cur_label} Mean", "dash"),
        ]:
            if len(arr) > 0:
                stat = np.mean(arr)
                fig.add_vline(
                    x=stat,
                    line=dict(color=color, dash=dash, width=2),
                    # Move label to legend by using a dummy invisible trace
                )
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=color, dash=dash, width=2),
                        name=label,
                        showlegend=True,
                    )
                )
        fig.update_layout(
            title=feature_name,
            xaxis_title=feature_name,
            yaxis_title="Probability Density",
            legend_title="Legend",
            template="plotly_white",
        )
        if show_score:
            if normalize and show_raw_score and raw_score is not None:
                score_text = (
                    f"Drift Score ({feature_name}): <b>{drift_score:.3f}</b> (Raw: {raw_score:.3f})"
                    if drift_score is not None
                    else "Drift Score: N/A (empty input)"
                )
            else:
                score_text = (
                    f"Drift Score ({feature_name}): <b>{drift_score:.3f}</b>"
                    if drift_score is not None
                    else "Drift Score: N/A (empty input)"
                )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.13,
                showarrow=False,
                text=score_text,
                font=dict(size=16, color="black"),
                align="center",
                bgcolor="rgba(255,255,255,0.7)",
            )
        return fig

    @staticmethod
    def plot_feature_drift_mp(
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
        """Plot distribution drift for a single feature using Matplotlib.

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
        feature_name = feature_name or str(reference.name) if reference.name is not None else "feature"
        drift_score = None
        raw_score = None

        if len(reference) > 0 and len(current) > 0:
            # Import here to avoid circular imports
            from tab_right.drift.univariate import detect_univariate_drift_with_options

            # Get both raw and normalized scores
            result = detect_univariate_drift_with_options(
                reference, current, kind="continuous", normalize=normalize, normalization_method=normalization_method
            )

            drift_score = result["score"]
            if "raw_score" in result:
                raw_score = result["raw_score"]

        # Compute KDEs for smooth lines
        x_min = min(reference.min() if len(reference) else 0, current.min() if len(current) else 0)
        x_max = max(reference.max() if len(reference) else 1, current.max() if len(current) else 1)
        x_grid = np.linspace(x_min, x_max, 200)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot KDE for reference data
        if len(reference) > 1:
            kde_ref = gaussian_kde(reference)
            ax.plot(x_grid, kde_ref(x_grid), color="blue", label=ref_label)

            # Add vertical line for mean
            if len(reference) > 0:
                ref_mean = np.mean(reference)
                ax.axvline(ref_mean, color="blue", linestyle="--", label=f"{ref_label} Mean")

        # Plot KDE for current data
        if len(current) > 1:
            kde_cur = gaussian_kde(current)
            ax.plot(x_grid, kde_cur(x_grid), color="orange", label=cur_label)

            # Add vertical line for mean
            if len(current) > 0:
                cur_mean = np.mean(current)
                ax.axvline(cur_mean, color="orange", linestyle="--", label=f"{cur_label} Mean")

        # Add title and labels
        ax.set_title(feature_name)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Probability Density")
        ax.legend(title="Legend")

        # Add drift score as text annotation
        if show_score:
            if normalize and show_raw_score and raw_score is not None:
                score_text = (
                    f"Drift Score ({feature_name}): {drift_score:.3f} (Raw: {raw_score:.3f})"
                    if drift_score is not None
                    else "Drift Score: N/A (empty input)"
                )
            else:
                score_text = (
                    f"Drift Score ({feature_name}): {drift_score:.3f}"
                    if drift_score is not None
                    else "Drift Score: N/A (empty input)"
                )

            ax.annotate(
                score_text,
                xy=(0.5, 1.05),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        plt.tight_layout()

        return fig
