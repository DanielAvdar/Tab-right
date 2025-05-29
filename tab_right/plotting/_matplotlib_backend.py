"""Matplotlib backend for drift plotting."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plotting_utils import add_metrics_textbox, create_empty_figure, parse_bin_edges


def plot_multiple_features(
    drift_results: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    sort_by: str = "score",
    ascending: bool = False,
    top_n: Optional[int] = None,
    threshold: Optional[float] = None,
) -> plt.Figure:
    """Create a bar chart visualization of drift across multiple features.

    Args:
        drift_results: DataFrame with drift results.
        figsize: Figure size as (width, height) in inches.
        sort_by: Column to sort results by (usually "score").
        ascending: Whether to sort in ascending order.
        top_n: If specified, only show the top N features.
        threshold: If specified, mark features above this threshold in a different color.

    Returns:
        A matplotlib Figure object containing the generated plot.

    """
    if drift_results.empty:
        return create_empty_figure(figsize, "No drift data to plot.")

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
        ax.axvline(
            threshold,
            color="grey",
            linestyle="--",
            label=f"Threshold = {threshold:.2f}",
        )
        ax.legend()

    plt.tight_layout()
    return fig


def plot_continuous_feature(
    feature_density: pd.DataFrame,
    bins_or_cats: np.ndarray,
    ref_density: np.ndarray,
    cur_density: np.ndarray,
    column: str,
    figsize: Tuple[int, int] = (10, 6),
    show_metrics: bool = True,
    drift_metrics: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """Plot continuous feature distribution.

    Args:
        feature_density: DataFrame with feature density data.
        bins_or_cats: Array of bin labels.
        ref_density: Reference density values.
        cur_density: Current density values.
        column: Column name.
        figsize: Figure size as (width, height) in inches.
        show_metrics: Whether to show drift metrics in the plot.
        drift_metrics: DataFrame with drift metrics.

    Returns:
        A matplotlib Figure object containing the generated plot.

    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure densities are np.ndarray
    ref_density = np.asarray(ref_density)
    cur_density = np.asarray(cur_density)

    # Attempt to extract bin edges for plotting histogram-like bars
    try:
        bin_edges, centers = parse_bin_edges(bins_or_cats)
        widths = np.diff(bin_edges)
        ref_values = ref_density.tolist()
        cur_values = cur_density.tolist()
        ax.bar(
            centers,
            ref_values,
            width=widths,
            label="Reference",
            alpha=0.7,
            align="center",
        )
        ax.bar(
            centers,
            cur_values,
            width=widths,
            label="Current",
            alpha=0.7,
            align="center",
        )
    except ValueError:
        # Fallback if bin parsing fails (e.g., unexpected format)
        x = np.arange(len(bins_or_cats))
        ref_values = ref_density.tolist()
        cur_values = cur_density.tolist()
        ax.plot(x, ref_values, label="Reference", marker="o")
        ax.plot(x, cur_values, label="Current", marker="x")
        ax.set_xticks(x)
        ax.set_xticklabels(bins_or_cats, rotation=45, ha="right")  # Use bin labels directly

    ax.set_ylabel("Probability Mass")
    ax.set_xlabel("Bins")
    ax.set_title(f"Continuous Distribution Comparison: {column}")
    ax.legend()

    # Add metrics if available
    if show_metrics and drift_metrics is not None and not drift_metrics.empty:
        add_metrics_textbox(ax, drift_metrics.iloc[0])

    plt.tight_layout()
    return fig


def plot_categorical_feature(
    feature_density: pd.DataFrame,
    bins_or_cats: np.ndarray,
    ref_density: np.ndarray,
    cur_density: np.ndarray,
    column: str,
    figsize: Tuple[int, int] = (10, 6),
    show_metrics: bool = True,
    drift_metrics: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """Plot categorical feature distribution.

    Args:
        feature_density: DataFrame with feature density data.
        bins_or_cats: Array of category names.
        ref_density: Reference density values.
        cur_density: Current density values.
        column: Column name.
        figsize: Figure size as (width, height) in inches.
        show_metrics: Whether to show drift metrics in the plot.
        drift_metrics: DataFrame with drift metrics.

    Returns:
        A matplotlib Figure object containing the generated plot.

    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure densities are np.ndarray
    ref_density = np.asarray(ref_density)
    cur_density = np.asarray(cur_density)

    x = np.arange(len(bins_or_cats))
    width = 0.35
    ref_values = ref_density.tolist()
    cur_values = cur_density.tolist()
    ax.bar(x - width / 2, ref_values, width, label="Reference", alpha=0.7)
    ax.bar(x + width / 2, cur_values, width, label="Current", alpha=0.7)
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(bins_or_cats, rotation=45, ha="right")
    ax.set_title(f"Categorical Distribution Comparison: {column}")
    ax.legend()

    # Add metrics if available
    if show_metrics and drift_metrics is not None and not drift_metrics.empty:
        add_metrics_textbox(ax, drift_metrics.iloc[0])

    plt.tight_layout()
    return fig


def plot_drift_values(
    drift_df: pd.DataFrame,
    value_col: str = "value",
    feature_col: str = "feature",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot drift values for each feature as a bar chart using Matplotlib.

    Args:
        drift_df: DataFrame with drift results. Should contain columns for feature names and drift values.
        value_col: Name of the column containing drift values.
        feature_col: Name of the column containing feature names.
        figsize: Figure size as (width, height) in inches.

    Returns:
        plt.Figure: Matplotlib figure with bar chart of drift values by feature.

    """
    drift_df_sorted = drift_df.sort_values(value_col, ascending=False)

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        drift_df_sorted[feature_col],
        drift_df_sorted[value_col],
        color="indianred",
    )

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Customize plot
    ax.set_title("Univariate Drift by Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Drift Value")
    plt.xticks(rotation=-45, ha="left")
    plt.tight_layout()

    return fig
