import pandas as pd
import plotly.graph_objects as go

from tab_right.plotting.plotly import plot_feature_drift


def test_plot_feature_drift_basic():
    ref = pd.Series([1.0, 2.0, 2.5, 3.0, 4.0])
    cur = pd.Series([1.5, 2.2, 2.8, 3.5, 4.2])
    fig = plot_feature_drift(ref, cur, feature_name="test_feature")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # two histograms
    assert fig.data[0].type == "histogram"
    assert fig.data[1].type == "histogram"
    # Check annotation for drift score
    assert any("Drift Score" in (a.text or "") for a in fig.layout.annotations or [])


def test_plot_feature_drift_empty():
    ref = pd.Series([], dtype=float)
    cur = pd.Series([], dtype=float)
    fig = plot_feature_drift(ref, cur, feature_name="empty_feature")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
