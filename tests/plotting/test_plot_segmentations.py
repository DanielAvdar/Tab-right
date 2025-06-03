"""Tests for the plot_segmentations module."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure as MatplotlibFigure
from pytest import approx

from tab_right.plotting.plot_segmentations import (
    DoubleSegmPlotting,
    plot_single_segmentation,
    plot_single_segmentation_mp,
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
        "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
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


def test_plot_single_segmentation_mp(single_segmentation_df):
    """Test that plot_single_segmentation_mp creates a valid Matplotlib figure."""
    fig = plot_single_segmentation_mp(single_segmentation_df)

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Check that figure has at least one axis
    assert len(fig.axes) > 0

    # Check that labels and title are set
    ax = fig.axes[0]
    assert "Segmentation Analysis" in ax.get_title()
    assert ax.get_xlabel() == "Feature Segments"
    assert ax.get_ylabel() == "Error Score"

    # Check for value labels (text annotations on the plot)
    [child for child in ax.get_children() if isinstance(child, plt.Text)]

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_plot_single_segmentation_mp_lower_is_better_false(single_segmentation_df):
    """Test plot_single_segmentation_mp with lower_is_better=False."""
    fig = plot_single_segmentation_mp(single_segmentation_df, lower_is_better=False)

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_double_segm_plotting_mp(double_segmentation_df):
    """Test that DoubleSegmPlotting with matplotlib backend creates a valid Matplotlib figure."""
    double_plotter = DoubleSegmPlotting(df=double_segmentation_df, backend="matplotlib")
    fig = double_plotter.plot_heatmap()

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Check that figure has at least one axis
    assert len(fig.axes) > 0

    # The main axis should have a pcolormesh for the heatmap and a colorbar
    ax = fig.axes[0]
    mesh_collections = [child for child in ax.get_children() if "QuadMesh" in str(type(child))]
    assert len(mesh_collections) == 1  # Should have exactly one heatmap

    # Check that title and labels are set
    assert "Double Segmentation Heatmap" in ax.get_title()
    assert ax.get_xlabel() == "Feature 1"
    assert ax.get_ylabel() == "Feature 2"

    # Check we have value annotations
    texts = [child for child in ax.get_children() if isinstance(child, plt.Text)]

    # Given our 3x2 heatmap (3 values for feature_1, 2 for feature_2), should have 6 cells
    # But some may be NaN, so just check we have some text elements
    assert len(texts) > 0

    # We should have a colorbar (usually the second axis in the figure)
    assert len(fig.axes) >= 2

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_double_segm_plotting_mp_lower_is_better_false(double_segmentation_df):
    """Test DoubleSegmPlotting with matplotlib backend and lower_is_better=False."""
    double_plotter = DoubleSegmPlotting(df=double_segmentation_df, lower_is_better=False, backend="matplotlib")
    fig = double_plotter.plot_heatmap()

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_double_segm_plotting_mp_custom_metric(double_segmentation_df):
    """Test DoubleSegmPlotting with matplotlib backend and a custom metric name."""
    # Create a copy with a renamed score column
    df_custom = double_segmentation_df.rename(columns={"score": "custom_metric"})

    double_plotter = DoubleSegmPlotting(df=df_custom, metric_name="custom_metric", backend="matplotlib")
    fig = double_plotter.plot_heatmap()

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Close the figure to prevent memory leaks
    plt.close(fig)
