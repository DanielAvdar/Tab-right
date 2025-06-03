import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure as MatplotlibFigure
from plotly import graph_objects as go

from tab_right.plotting import DriftPlotter


def test_plot_drift_basic():
    # Create a sample DataFrame with drift values
    drift_df = pd.DataFrame({"feature": ["feature1", "feature2", "feature3"], "value": [0.8, 0.5, 0.2]})

    # Test the function using DriftPlotter.plot_drift
    fig = DriftPlotter.plot_drift(None, drift_df)

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

    # Test with custom column names using DriftPlotter.plot_drift
    fig = DriftPlotter.plot_drift(None, drift_df, value_col="drift_score", feature_col="feat_name")

    # Verify output
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

    # Check data is sorted by drift_score
    assert list(fig.data[0].x) == ["A", "B", "C"]
    assert list(fig.data[0].y) == [0.9, 0.6, 0.3]


def test_plot_drift_mp_basic():
    # Create a sample DataFrame with drift values
    drift_df = pd.DataFrame({"feature": ["feature1", "feature2", "feature3"], "value": [0.8, 0.5, 0.2]})

    # Test the function using DriftPlotter.plot_drift_mp
    fig = DriftPlotter.plot_drift_mp(None, drift_df)

    # Verify the output is a matplotlib Figure
    assert isinstance(fig, MatplotlibFigure)

    # Check that figure has at least one axis
    assert len(fig.axes) > 0

    # Check the title and labels
    ax = fig.axes[0]
    assert "Univariate Drift by Feature" in ax.get_title()
    assert ax.get_xlabel() == "Feature"
    assert ax.get_ylabel() == "Drift Value"

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_plot_drift_mp_custom_columns():
    # Create a sample DataFrame with custom column names
    drift_df = pd.DataFrame({"feat_name": ["A", "B", "C"], "drift_score": [0.9, 0.6, 0.3]})

    # Test with custom column names using DriftPlotter.plot_drift_mp
    fig = DriftPlotter.plot_drift_mp(None, drift_df, value_col="drift_score", feature_col="feat_name")

    # Verify the output is a matplotlib Figure
    assert isinstance(fig, MatplotlibFigure)

    # Check that figure has at least one axis
    assert len(fig.axes) > 0

    # Check there are appropriate number of bars (3 for A, B, C)
    ax = fig.axes[0]

    # Check for value labels (text annotations on the plot)
    texts = [child for child in ax.get_children() if isinstance(child, plt.Text)]
    visible_texts = [text for text in texts if text.get_position()[1] > 0]

    # Should have at least 3 value labels (one for each bar)
    assert len(visible_texts) >= 3

    # Close the figure to prevent memory leaks
    plt.close(fig)
