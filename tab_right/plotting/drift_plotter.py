"""Implementation of the DriftPlotP protocol using Matplotlib."""

from typing import Any, Dict, Iterable, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tab_right.base_architecture.drift_plot_protocols import DriftPlotP
from tab_right.base_architecture.drift_protocols import DriftCalcP


class DriftPlotter(DriftPlotP):
    """Implementation of DriftPlotP using Matplotlib."""

    def __init__(self, drift_calc: DriftCalcP):
        """Initialize the DriftPlotter with a drift calculator.

        Args:
            drift_calc: An instance of DriftCalcP that will provide drift metrics and
                probability densities for plotting.

        Raises:
            TypeError: If drift_calc is not an instance of DriftCalcP.

        """
        if not isinstance(drift_calc, DriftCalcP):
            raise TypeError("drift_calc must be an instance of DriftCalcP")
        self.drift_calc = drift_calc
        # Add a _feature_types attribute that can be used by mypy
        self._feature_types = getattr(drift_calc, "_feature_types", {})

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

        if drift_results.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No drift data to plot.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        # Sort and filter
        drift_results = drift_results.sort_values(by=sort_by, ascending=ascending)
        if top_n:
            drift_results = drift_results.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        features = drift_results["feature"]
        scores = drift_results["score"]
        colors = ["red" if threshold is not None and score >= threshold else "blue" for score in scores]

        bars = ax.barh(features, scores, color=colors)
        ax.set_xlabel("Drift Score (Type Varies by Feature)")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Drift Scores")
        ax.invert_yaxis()  # Highest score on top

        # Add score labels
        ax.bar_label(bars, fmt="%.3f", padding=3)

        if threshold is not None:
            ax.axvline(threshold, color="grey", linestyle="--", label=f"Threshold = {threshold:.2f}")
            ax.legend()

        plt.tight_layout()
        return fig

    def plot_single(
        self, column: str, bins: int = 10, figsize: Tuple[int, int] = (10, 6), show_metrics: bool = True, **kwargs: Any
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

        Raises:
            ValueError: If the column is not found or its type is not determined.

        """
        if column not in self._feature_types:
            raise ValueError(f"Column '{column}' not found or type not determined.")

        col_type = self._feature_types[column]
        density_df = self.drift_calc.get_prob_density(columns=[column], bins=bins)
        drift_metrics = self.drift_calc(columns=[column], bins=bins)

        fig, ax = plt.subplots(figsize=figsize)

        if density_df.empty:
            ax.text(0.5, 0.5, f"No data available for column '{column}'.", ha="center", va="center")
            return fig

        feature_density = density_df[density_df["feature"] == column]
        bins_or_cats = feature_density["bin"].values
        ref_density = feature_density["ref_density"].values
        cur_density = feature_density["cur_density"].values

        if col_type == "categorical":
            x = np.arange(len(bins_or_cats))
            width = 0.35
            # Cast numpy arrays to avoid type issues with bar function
            ref_values = cast(np.ndarray, ref_density).tolist()
            cur_values = cast(np.ndarray, cur_density).tolist()
            ax.bar(x - width / 2, ref_values, width, label="Reference", alpha=0.7)
            ax.bar(x + width / 2, cur_values, width, label="Current", alpha=0.7)
            ax.set_ylabel("Proportion")
            ax.set_xticks(x)
            ax.set_xticklabels(bins_or_cats, rotation=45, ha="right")
            ax.set_title(f"Categorical Distribution Comparison: {column}")
        elif col_type == "continuous":
            # Attempt to extract bin edges for plotting histogram-like bars
            try:
                bin_edges_str = [s.strip("()[]") for s in bins_or_cats]
                bin_edges = sorted(list(set([float(edge) for item in bin_edges_str for edge in item.split("-")])))
                widths = np.diff(bin_edges)
                centers = bin_edges[:-1] + widths / 2
                # Cast numpy arrays to avoid type issues with bar function
                ref_values = cast(np.ndarray, ref_density).tolist()
                cur_values = cast(np.ndarray, cur_density).tolist()
                ax.bar(centers, ref_values, width=widths, label="Reference", alpha=0.7, align="center")
                ax.bar(centers, cur_values, width=widths, label="Current", alpha=0.7, align="center")
            except Exception:  # Catch specific exceptions when possible
                # Fallback if bin parsing fails (e.g., unexpected format)
                x = np.arange(len(bins_or_cats))
                # Cast numpy arrays to avoid type issues with plot function
                ref_values = cast(np.ndarray, ref_density).tolist()
                cur_values = cast(np.ndarray, cur_density).tolist()
                ax.plot(x, ref_values, label="Reference", marker="o")
                ax.plot(x, cur_values, label="Current", marker="x")
                ax.set_xticks(x)
                ax.set_xticklabels(bins_or_cats, rotation=45, ha="right")  # Use bin labels directly

            ax.set_ylabel("Probability Mass")  # Since we multiplied density by bin width
            ax.set_xlabel("Bins")
            ax.set_title(f"Continuous Distribution Comparison: {column}")

        ax.legend()

        if show_metrics and not drift_metrics.empty:
            metric_info = drift_metrics.iloc[0]
            score = metric_info["score"]
            metric_type = metric_info["type"]
            raw_score = metric_info["raw_score"]
            # Use raw_score for display if different and available
            display_score = raw_score if pd.notna(raw_score) and raw_score != score else score
            metrics_text = f"{metric_type.replace('_', ' ').title()}: {display_score:.4f}"
            # Add text box with metrics
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)

        plt.tight_layout()
        return fig

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
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, f"Error plotting {col}", ha="center", va="center")
                plots[col] = fig

        # Store the figures but don't close them yet - they're still needed for return
        result = {k: fig for k, fig in plots.items()}

        # Close all figures including any that might have been created internally
        plt.close("all")

        return result
