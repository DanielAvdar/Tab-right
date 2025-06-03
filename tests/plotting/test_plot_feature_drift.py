import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure as MatplotlibFigure
from plotly import graph_objects as go

from tab_right.plotting import DriftPlotter


def test_plot_feature_drift_basic():
    ref = pd.Series([1.0, 2.0, 2.5, 3.0, 4.0])
    cur = pd.Series([1.5, 2.2, 2.8, 3.5, 4.2])
    fig = DriftPlotter.plot_feature_drift(ref, cur, feature_name="test_feature")
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
    fig = DriftPlotter.plot_feature_drift(ref, cur, feature_name="empty_feature")
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
    fig = DriftPlotter.plot_feature_drift(
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


def test_plot_feature_drift_mp_basic():
    ref = pd.Series([1.0, 2.0, 2.5, 3.0, 4.0])
    cur = pd.Series([1.5, 2.2, 2.8, 3.5, 4.2])
    fig = DriftPlotter.plot_feature_drift_mp(ref, cur, feature_name="test_feature")

    # Verify the output is a matplotlib Figure
    assert isinstance(fig, MatplotlibFigure)

    # Check that figure has at least one axis
    assert len(fig.axes) > 0

    # The main axis should have at least 2 lines (for KDEs) and 2 vertical lines (for means)
    ax = fig.axes[0]
    lines = [child for child in ax.get_children() if isinstance(child, plt.Line2D)]
    assert len(lines) >= 4

    # Check that labels are set
    assert ax.get_xlabel() is not None
    assert ax.get_ylabel() == "Probability Density"

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_plot_feature_drift_mp_empty():
    ref = pd.Series([], dtype=float)
    cur = pd.Series([], dtype=float)
    fig = DriftPlotter.plot_feature_drift_mp(ref, cur, feature_name="empty_feature")

    # Verify the output is a matplotlib Figure
    assert isinstance(fig, MatplotlibFigure)

    # Check that figure has at least one axis
    assert len(fig.axes) > 0

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_plot_feature_drift_mp_with_raw_scores():
    """Test the plot_feature_drift_mp function with raw score display option."""
    # Create test data
    ref = pd.Series([1, 2, 3, 4, 5])
    cur = pd.Series([2, 3, 4, 5, 6])

    # Test with show_raw_score=True
    fig = DriftPlotter.plot_feature_drift_mp(
        reference=ref,
        current=cur,
        feature_name="test_feature",
        show_score=True,
        normalize=True,
        show_raw_score=True,
    )

    # Verify the output is a matplotlib Figure
    assert isinstance(fig, MatplotlibFigure)

    # Check that figure has at least one axis
    assert len(fig.axes) > 0

    # Check for text annotations (harder to inspect directly in matplotlib)
    ax = fig.axes[0]
    texts = [child for child in ax.get_children() if isinstance(child, plt.Text)]
    annotation_texts = [text.get_text() for text in texts if hasattr(text, "get_text")]

    # Check if any annotation contains 'Raw:'
    assert any("Raw:" in text for text in annotation_texts if isinstance(text, str))

    # Close the figure to prevent memory leaks
    plt.close(fig)
