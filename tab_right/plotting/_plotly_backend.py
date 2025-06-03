"""Plotly backend for drift plotting."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_drift_values(
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
    drift_df_sorted = drift_df.sort_values(value_col, ascending=False)
    fig = go.Figure(
        go.Bar(
            x=drift_df_sorted[feature_col],
            y=drift_df_sorted[value_col],
            marker_color="indianred",
            name="Drift Value",
        )
    )
    fig.update_layout(
        title="Univariate Drift by Feature",
        xaxis_title="Feature",
        yaxis_title="Drift Value",
        xaxis_tickangle=-45,
    )
    return fig


def plot_feature_drift_plotly(
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
        go.Figure: Plotly figure with overlaid histograms, means, medians, and drift score annotation.

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
