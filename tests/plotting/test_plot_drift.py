import pandas as pd
from plotly import graph_objects as go

from tab_right.plotting import plot_drift


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
