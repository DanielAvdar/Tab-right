"""Module for plotting distribution drift for a single feature."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly import graph_objects as go


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
    """Plot distribution drift for a single feature, with means, medians, and drift score (Earth Mover's Distance).

    Parameters
    ----------
    reference : pd.Series
        Reference (train) data for the feature.
    current : pd.Series
        Current (test) data for the feature.
    feature_name : str, optional
        Name of the feature (for labeling plots).
    show_score : bool, default True
        Whether to display the drift score annotation.
    ref_label : str, default "Train Dataset"
        Label for the reference data.
    cur_label : str, default "Test Dataset"
        Label for the current data.
    normalize : bool, default True
        Whether to normalize the Wasserstein distance.
    normalization_method : str, default "range"
        Method to use for normalization: "range", "std", or "iqr".
    show_raw_score : bool, default False
        Whether to show both normalized and raw scores.

    Returns
    -------
    go.Figure
        Plotly figure with overlaid histograms, means, medians, and drift score annotation.

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
    from scipy.stats import gaussian_kde

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
) -> Figure:
    """Plot distribution drift for a single feature using matplotlib.

    Parameters
    ----------
    reference : pd.Series
        Reference (train) data for the feature.
    current : pd.Series
        Current (test) data for the feature.
    feature_name : str, optional
        Name of the feature (for labeling plots).
    show_score : bool, default True
        Whether to display the drift score annotation.
    ref_label : str, default "Train Dataset"
        Label for the reference data.
    cur_label : str, default "Test Dataset"
        Label for the current data.
    normalize : bool, default True
        Whether to normalize the Wasserstein distance.
    normalization_method : str, default "range"
        Method to use for normalization: "range", "std", or "iqr".
    show_raw_score : bool, default False
        Whether to show both normalized and raw scores.

    Returns
    -------
    Figure
        Matplotlib figure with overlaid distributions, means, and drift score.

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
    from scipy.stats import gaussian_kde

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
