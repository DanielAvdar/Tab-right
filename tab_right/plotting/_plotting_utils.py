"""Utility functions for drift plotting."""

from typing import Any, Dict, Iterable, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def validate_column_exists(column: str, feature_types: Dict[str, str]) -> None:
    """Validate that the column exists in the feature types dictionary.

    Args:
        column: The column name to validate.
        feature_types: Dictionary mapping column names to their types.

    Raises:
        ValueError: If the column is not found or its type is not determined.
    """
    if column not in feature_types:
        raise ValueError(f"Column '{column}' not found or type not determined.")


def create_empty_figure(figsize: Tuple[int, int] = (10, 6), message: str = "No data to plot.") -> plt.Figure:
    """Create an empty figure with a message.

    Args:
        figsize: Figure size as (width, height) in inches.
        message: The message to display in the figure.

    Returns:
        A matplotlib Figure object containing the generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def add_metrics_textbox(
    ax: plt.Axes, metric_info: pd.Series, position: Tuple[float, float] = (0.05, 0.95)
) -> None:
    """Add a metrics textbox to the plot.

    Args:
        ax: The matplotlib Axes object to add the textbox to.
        metric_info: A pandas Series containing the metrics.
        position: The position of the textbox as (x, y) in axes coordinates.
    """
    score = metric_info["score"]
    metric_type = metric_info["type"]
    raw_score = metric_info["raw_score"]
    # Use raw_score for display if different and available
    display_score = raw_score if pd.notna(raw_score) and raw_score != score else score
    metrics_text = f"{metric_type.replace('_', ' ').title()}: {display_score:.4f}"
    # Add text box with metrics
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        position[0],
        position[1],
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )


def parse_bin_edges(bins_or_cats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parse bin edges from bin labels.

    Args:
        bins_or_cats: Array of bin labels.

    Returns:
        A tuple containing (bin_edges, bin_centers).

    Raises:
        ValueError: If the bin labels cannot be parsed.
    """
    try:
        bin_edges_str = [s.strip("()[]") for s in bins_or_cats]
        bin_edges = sorted(list(set([float(edge) for item in bin_edges_str for edge in item.split("-")])))
        widths = np.diff(bin_edges)
        centers = bin_edges[:-1] + widths / 2
        return np.array(bin_edges), np.array(centers)
    except Exception:
        raise ValueError("Could not parse bin edges from bin labels.")