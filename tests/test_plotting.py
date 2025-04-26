import pandas as pd
from tab_right.plotting.plotly import plot_segmentations, plot_segmentation_heatmap
import plotly.graph_objects as go
import pytest
@pytest.mark.skip
def test_plot_segmentations_basic():
    df = pd.DataFrame({
        "segment": ["A", "B", "C"],
        "score": [0.8, 0.5, 0.9],
    })
    fig = plot_segmentations(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
@pytest.mark.skip
def test_plot_segmentations_ascending():
    df = pd.DataFrame({
        "segment": ["A", "B", "C"],
        "score": [0.2, 0.5, 0.1],
    })
    fig = plot_segmentations(df, ascending=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

def test_plot_segmentations_single_segment():
    df = pd.DataFrame({
        "segment": ["A"],
        "score": [0.7],
    })
    fig = plot_segmentations(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    fig.show()
