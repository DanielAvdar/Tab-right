"""Module for plotting distribution drift for a single feature."""

import pandas as pd
import plotly.graph_objects as go


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
    reference:
        Reference (training) dataset.
    current:
        Current (test) dataset.
    feature_name:
        Name of the feature to be plotted. If None, the name will be taken from the Series.
    show_score:
        Whether to show the drift score in the title.
    ref_label:
        Label for the reference dataset.
    cur_label:
        Label for the current dataset.

    Returns
    -------
        Plotly Figure object.

    """
