"""Module for plotting distribution drift for a single feature."""

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from scipy.stats import wasserstein_distance


def plot_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str = None,
    show_score: bool = True,
    ref_label: str = "Train Dataset",
    cur_label: str = "Test Dataset",
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

    Returns
    -------
    go.Figure
        Plotly figure with overlaid histograms, means, medians, and drift score annotation.

    """
    feature_name = feature_name or reference.name or "feature"
    drift_score = None
    if len(reference) > 0 and len(current) > 0:
        drift_score = wasserstein_distance(reference, current)
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
                line=dict(color="navy"),
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
                line=dict(color="mediumaquamarine"),
            )
        )
    # Means and medians
    for arr, color, label, dash in [
        (reference, "navy", f"{ref_label} Mean", "dash"),
        (current, "mediumaquamarine", f"{cur_label} Mean", "dash"),
        (reference, "navy", f"{ref_label} Median", "dot"),
        (current, "mediumaquamarine", f"{cur_label} Median", "dot"),
    ]:
        if len(arr) > 0:
            stat = np.mean(arr) if "Mean" in label else np.median(arr)
            fig.add_vline(
                x=stat,
                line=dict(color=color, dash=dash, width=2),
                annotation_text=label,
                annotation_position="top",
                annotation_font_color=color,
                annotation_font_size=12,
            )
    fig.update_layout(
        title=feature_name,
        xaxis_title=feature_name,
        yaxis_title="Probability Density",
        legend_title="Legend",
        template="plotly_white",
    )
    if show_score:
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
