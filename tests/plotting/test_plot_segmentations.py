"""Tests for the plot_segmentations module."""

import pandas as pd
import plotly.graph_objects as go
import pytest
from pytest import approx

from tab_right.base_architecture.seg_protocols_check import CheckDoubleSegmPlotting
from tab_right.plotting.plot_segmentations import (
    plot_single_segmentation,
)


@pytest.fixture
def single_segmentation_df():
    """Create a sample DataFrame for testing single segmentation plotting."""
    return pd.DataFrame({
        "segment_id": [1, 2, 3, 4],
        "segment_name": ["A <= 10", "10 < A <= 20", "20 < A <= 30", "A > 30"],
        "score": [0.1, 0.25, 0.15, 0.3],
    })


def test_plot_single_segmentation(single_segmentation_df):
    """Test that plot_single_segmentation creates a valid Plotly figure."""
    fig = plot_single_segmentation(single_segmentation_df)

    # Check that we got a plotly figure
    assert isinstance(fig, go.Figure)

    # Check that the figure contains a bar trace
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)

    # Check that the data was correctly passed to the plot
    assert len(fig.data[0].x) == len(single_segmentation_df)
    assert list(fig.data[0].y) == approx(list(single_segmentation_df["score"]))


class TestDoubleSegmPlotting(CheckDoubleSegmPlotting):
    """Test class for double segmentation plotting."""
