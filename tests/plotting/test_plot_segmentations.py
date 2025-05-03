"""Tests for the plot_segmentations module."""

import pandas as pd
import plotly.graph_objects as go
import pytest
from pytest import approx

from tab_right.plotting.plot_segmentations import (
    plot_double_segmentation,
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


@pytest.fixture
def double_segmentation_df():
    """Create a sample DataFrame for testing double segmentation plotting."""
    return pd.DataFrame({
        "segment_id": [1, 2, 3, 4, 5, 6],
        "feature_1": ["A <= 10", "A <= 10", "10 < A <= 20", "10 < A <= 20", "A > 20", "A > 20"],
        "feature_2": ["B <= 5", "B > 5", "B <= 5", "B > 5", "B <= 5", "B > 5"],
        "score": [0.1, 0.25, 0.15, 0.3, 0.2, 0.35],
    })


@pytest.fixture
def mock_dt_segmentation(single_segmentation_df):
    """Create a mock DecisionTreeSegmentation object."""

    class MockDTSegmentation:
        def __init__(self, df, feature_names=None):
            self.df = df
            self.feature_names = feature_names or ["feature"]

        def get_segment_df(self):
            return self.df

        def get_segment_stats(self, n_segments=3):
            stats = pd.DataFrame({
                "segment_id": self.df["segment_id"].iloc[:n_segments],
                "count": [100, 80, 60][:n_segments],
                "mean_error": self.df["score"].iloc[:n_segments],
                "std_error": [0.05, 0.07, 0.04][:n_segments],
            })
            return stats

    return MockDTSegmentation(single_segmentation_df)


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


def test_plot_double_segmentation(double_segmentation_df):
    """Test that plot_double_segmentation creates a valid Plotly figure."""
    fig = plot_double_segmentation(double_segmentation_df)

    # Check that we got a plotly figure
    assert isinstance(fig, go.Figure)

    # Check that the figure contains a heatmap trace
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)

    # Check that layout contains proper titles
    assert "Feature 1" in fig.layout.xaxis.title.text
    assert "Feature 2" in fig.layout.yaxis.title.text
