from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tab_right.plotting import plot_feature_drift
from tab_right.plotting.plot_drift import plot_drift
from tab_right.plotting.plot_segmentations import (
    plot_dt_segmentation,
    plot_dt_segmentation_with_stats,
    plot_segmentations,
)


def test_plot_feature_drift_basic():
    ref = pd.Series([1.0, 2.0, 2.5, 3.0, 4.0])
    cur = pd.Series([1.5, 2.2, 2.8, 3.5, 4.2])
    fig = plot_feature_drift(ref, cur, feature_name="test_feature")
    assert isinstance(fig, go.Figure)
    # Should have two KDE lines (scatter) and two mean lines (dummy traces for legend)
    assert len(fig.data) == 4
    # The first two traces are KDE lines
    assert all(trace.type == "scatter" for trace in fig.data)
    assert fig.data[0].name == "Train Dataset"
    assert fig.data[1].name == "Test Dataset"
    # The next two traces are dummy mean lines for legend
    assert fig.data[2].name == "Train Dataset Mean"
    assert fig.data[3].name == "Test Dataset Mean"
    # Check annotation for drift score
    assert any("Drift Score" in (a.text or "") for a in fig.layout.annotations or [])


def test_plot_feature_drift_empty():
    ref = pd.Series([], dtype=float)
    cur = pd.Series([], dtype=float)
    fig = plot_feature_drift(ref, cur, feature_name="empty_feature")
    assert isinstance(fig, go.Figure)
    # Should have no KDE lines if both are empty
    assert len(fig.data) == 0
    # Should still have annotation for drift score
    assert any("Drift Score" in (a.text or "") for a in fig.layout.annotations or [])


def test_plot_feature_drift_with_raw_scores():
    """Test the plot_feature_drift function with raw score display option enabled."""
    # Create test data
    ref = pd.Series([1, 2, 3, 4, 5])
    cur = pd.Series([2, 3, 4, 5, 6])

    # Test with show_raw_score=True to hit the uncovered line 125
    fig = plot_feature_drift(
        reference=ref,
        current=cur,
        feature_name="test_feature",
        show_score=True,
        normalize=True,
        show_raw_score=True,  # This should hit the uncovered line
    )

    # Verify that the figure was created successfully
    assert isinstance(fig, go.Figure)

    # Check that the annotation exists and contains both normalized and raw scores
    assert len(fig.layout.annotations) == 1
    annotation_text = fig.layout.annotations[0].text
    assert "Drift Score" in annotation_text
    assert "Raw:" in annotation_text  # This confirms the raw score is included


def test_plot_drift_basic():
    # Create a sample DataFrame with drift values
    drift_df = pd.DataFrame({"feature": ["feature1", "feature2", "feature3"], "value": [0.8, 0.5, 0.2]})

    # Test the function
    fig = plot_drift(drift_df)

    # Verify the output is a plotly Figure
    assert isinstance(fig, go.Figure)

    # Check there's a bar chart
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"

    # Check the data was sorted correctly
    assert list(fig.data[0].x) == ["feature1", "feature2", "feature3"]
    assert list(fig.data[0].y) == [0.8, 0.5, 0.2]

    # Check layout properties
    assert "Univariate Drift by Feature" in fig.layout.title.text
    assert fig.layout.xaxis.title.text == "Feature"
    assert fig.layout.yaxis.title.text == "Drift Value"


def test_plot_drift_custom_columns():
    # Create a sample DataFrame with custom column names
    drift_df = pd.DataFrame({"feat_name": ["A", "B", "C"], "drift_score": [0.9, 0.6, 0.3]})

    # Test with custom column names
    fig = plot_drift(drift_df, value_col="drift_score", feature_col="feat_name")

    # Verify output
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

    # Check data is sorted by drift_score
    assert list(fig.data[0].x) == ["A", "B", "C"]
    assert list(fig.data[0].y) == [0.9, 0.6, 0.3]


def test_plot_segmentations_basic():
    # Create a sample DataFrame with segment and score data
    df = pd.DataFrame({"segment": [1, 1, 1, 2, 2, 3, 3, 3], "score": [0.9, 0.8, 0.7, -0.5, -0.6, 0.2, 0.3, 0.1]})

    # Test with default parameters
    fig = plot_segmentations(df)

    # Verify the output is a plotly Figure
    assert isinstance(fig, go.Figure)

    # Should have 3 bar traces (one for each segment)
    assert len(fig.data) == 3

    # Check that each trace is a bar chart
    for trace in fig.data:
        assert trace.type == "bar"

    # Check segment names/colors
    assert fig.data[0].name == "1"  # First segment
    assert fig.data[1].name == "2"  # Second segment
    assert fig.data[2].name == "3"  # Third segment

    # Positive segments should be green, negative red
    assert fig.data[0].marker.color == "green"  # Segment 1 has positive scores
    assert fig.data[1].marker.color == "red"  # Segment 2 has negative scores
    assert fig.data[2].marker.color == "green"  # Segment 3 has positive scores

    # Check layout
    assert fig.layout.barmode == "group"


def test_plot_segmentations_custom_parameters():
    # Create a sample DataFrame with custom column names
    df = pd.DataFrame({"segment_id": [1, 1, 2, 2], "performance": [0.8, 0.9, -0.4, -0.3]})

    # Create an existing figure to update
    existing_fig = go.Figure()

    # Test with custom parameters
    fig = plot_segmentations(
        df,
        good_color="blue",
        bad_color="purple",
        score_col="performance",
        segment_col="segment_id",
        ascending=True,  # Sort ascending
        fig=existing_fig,
    )

    # Verify it's the same figure object that was passed in
    assert fig is existing_fig

    # Should have 2 bar traces
    assert len(fig.data) == 2

    # Note: The function sorts the entire dataframe by score values,
    # but when iterating through groups, it uses the first value in each group to determine color
    # The order of the segments in the figure depends on their order in the groupby iteration, not their score values
    for trace in fig.data:
        segment_id = int(trace.name)
        # Check color based on the first value for each segment in the original dataframe
        first_value = df[df["segment_id"] == segment_id]["performance"].iloc[0]
        expected_color = "blue" if first_value >= 0 else "purple"
        assert trace.marker.color == expected_color


@patch("plotly.graph_objects.Figure.show")
def test_plot_dt_segmentation_with_stats(mock_show):
    """Test the plot_dt_segmentation_with_stats function with a mocked segmentation."""
    # Create a mock DecisionTreeSegmentation
    mock_segmentation = Mock()
    mock_segmentation.tree_model = Mock()
    mock_segmentation.feature1_col = "feature1"
    mock_segmentation.feature2_col = "feature2"

    # Mock the get_feature_ranges method
    mock_segmentation.get_feature_ranges.return_value = [(0, 10), (0, 10)]

    # Mock the predict method on tree_model to return values for the grid
    mock_segmentation.tree_model.predict.return_value = np.ones(25)  # 5x5 grid when reshaped

    # Mock get_segment_stats to return some fake segment stats
    mock_segmentation.get_segment_stats.return_value = pd.DataFrame({
        "segment_id": [1, 2, 3],
        "mean_error": [0.8, 0.5, 0.3],
        "count": [10, 15, 20],
    })

    # Call the function with show=True to hit the fig.show() line
    fig = plot_dt_segmentation_with_stats(
        segmentation=mock_segmentation,
        resolution=5,  # Small value for faster test
        show=True,  # This should trigger the fig.show() call
    )

    # Verify the figure was created correctly
    assert isinstance(fig, go.Figure)

    # Verify show() was called
    mock_show.assert_called_once()

    # Verify subplots were created (should have two traces - heatmap and bar)
    assert len(fig.data) == 2
    assert fig.data[0].type == "heatmap"
    assert fig.data[1].type == "bar"


@patch("plotly.graph_objects.Figure.show")
def test_plot_dt_segmentation_show_true(mock_show):
    """Test the plot_dt_segmentation function with show=True."""
    # Create a mock DecisionTreeSegmentation
    mock_segmentation = Mock()
    mock_segmentation.tree_model = Mock()
    mock_segmentation.feature1_col = "feature1"
    mock_segmentation.feature2_col = "feature2"

    # Mock the get_feature_ranges method
    mock_segmentation.get_feature_ranges.return_value = [(0, 10), (0, 10)]

    # Mock the predict method on tree_model to return values for the grid
    mock_segmentation.tree_model.predict.return_value = np.ones(25)  # 5x5 grid when reshaped

    # Call the function with show=True to hit the fig.show() line
    fig = plot_dt_segmentation(
        segmentation=mock_segmentation,
        resolution=5,  # Small value for faster test
        show=True,  # This should trigger the fig.show() call
    )

    # Verify the figure was created correctly
    assert isinstance(fig, go.Figure)

    # Verify show() was called
    mock_show.assert_called_once()

    # Verify the figure has the expected trace
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
