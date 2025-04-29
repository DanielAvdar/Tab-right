"""Tests for decision tree segmentation feature."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from tab_right.plotting.plot_segmentations import plot_dt_segmentation, plot_dt_segmentation_with_stats
from tab_right.segmentations.dt_segmentation import DecisionTreeSegmentation


@pytest.fixture
def sample_data():
    """Create sample data for testing decision tree segmentation."""
    # Create a synthetic dataset with known patterns
    np.random.seed(42)
    n_samples = 500
    x = pd.DataFrame({
        "feature1": np.random.uniform(-2, 2, n_samples),
        "feature2": np.random.uniform(-2, 2, n_samples),
        "feature3": np.random.normal(0, 1, n_samples),
    })

    # Create a target with quadratic pattern and some noise
    noise = np.random.normal(0, 0.5, n_samples)
    y = x["feature1"] ** 2 - x["feature2"] ** 2 + 0.5 * x["feature3"] + noise

    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Train a model with some error patterns
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(x_train, y_train)

    # Generate predictions
    y_pred = model.predict(x_test)

    return {"x_test": x_test, "y_test": y_test, "y_pred": y_pred}


def test_dt_segmentation_initialization():
    """Test initialization of DecisionTreeSegmentation class."""
    dt_seg = DecisionTreeSegmentation()
    assert dt_seg.max_depth == 5
    assert dt_seg.min_samples_leaf == 20
    assert dt_seg.tree_model is None
    assert dt_seg.feature_1_col is None
    assert dt_seg.feature_2_col is None

    # Test with custom parameters
    dt_seg = DecisionTreeSegmentation(max_depth=3, min_samples_leaf=10)
    assert dt_seg.max_depth == 3
    assert dt_seg.min_samples_leaf == 10


def test_dt_segmentation_fit(sample_data):
    """Test fitting the decision tree segmentation model."""
    x_test = sample_data["x_test"]
    y_test = sample_data["y_test"]
    y_pred = sample_data["y_pred"]

    dt_seg = DecisionTreeSegmentation(max_depth=4)
    dt_seg.fit(x_test, y_test, y_pred, feature_names=["feature1", "feature2"])

    # Check if model was fitted
    assert dt_seg.tree_model is not None
    assert dt_seg.feature_1_col == "feature1"
    assert dt_seg.feature_2_col == "feature2"


def test_dt_segmentation_numpy_input():
    """Test fitting with numpy arrays instead of DataFrames."""
    np.random.seed(42)
    x = np.random.rand(100, 3)
    y_true = np.sin(x[:, 0]) + x[:, 1] ** 2
    y_pred = y_true + np.random.normal(0, 0.2, size=100)

    dt_seg = DecisionTreeSegmentation()
    dt_seg.fit(x, y_true, y_pred, feature_names=["Feature A", "Feature B"])

    assert dt_seg.tree_model is not None
    assert dt_seg.feature_1_col == "Feature A"
    assert dt_seg.feature_2_col == "Feature B"


def test_dt_segmentation_plotting_function(sample_data):
    """Test visualization of error segments using plotting function."""
    x_test = sample_data["x_test"]
    y_test = sample_data["y_test"]
    y_pred = sample_data["y_pred"]

    dt_seg = DecisionTreeSegmentation(max_depth=3)
    dt_seg.fit(x_test, y_test, y_pred, feature_names=["feature1", "feature2"])

    # Test plotting function
    fig = plot_dt_segmentation(dt_seg, show=False)
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) > 0
    assert fig.data[0].type == "heatmap"


def test_dt_segmentation_plotting_with_stats(sample_data):
    """Test combined visualization with stats."""
    x_test = sample_data["x_test"]
    y_test = sample_data["y_test"]
    y_pred = sample_data["y_pred"]

    dt_seg = DecisionTreeSegmentation(max_depth=3)
    dt_seg.fit(x_test, y_test, y_pred, feature_names=["feature1", "feature2"])

    # Test combined visualization function
    fig = plot_dt_segmentation_with_stats(dt_seg, show=False)
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) >= 2  # Should have at least 2 traces (heatmap and bar chart)
    assert fig.data[0].type == "heatmap"


def test_dt_segmentation_get_segment_df(sample_data):
    """Test segment DataFrame generation."""
    x_test = sample_data["x_test"]
    y_test = sample_data["y_test"]
    y_pred = sample_data["y_pred"]

    dt_seg = DecisionTreeSegmentation(max_depth=3)
    dt_seg.fit(x_test, y_test, y_pred, feature_names=["feature1", "feature2"])

    segment_df = dt_seg.get_segment_df(n_segments=5)

    assert isinstance(segment_df, pd.DataFrame)
    assert "segment_id" in segment_df.columns
    assert "mean_error" in segment_df.columns
    assert dt_seg.feature_1_col in segment_df.columns
    assert dt_seg.feature_2_col in segment_df.columns
    assert dt_seg.error_col in segment_df.columns


def test_dt_segmentation_get_segment_stats(sample_data):
    """Test segment statistics generation."""
    x_test = sample_data["x_test"]
    y_test = sample_data["y_test"]
    y_pred = sample_data["y_pred"]

    dt_seg = DecisionTreeSegmentation(max_depth=3)
    dt_seg.fit(x_test, y_test, y_pred, feature_names=["feature1", "feature2"])

    stats = dt_seg.get_segment_stats(n_segments=5)

    assert isinstance(stats, pd.DataFrame)
    assert len(stats) <= 5  # Might be fewer if there are fewer leaf nodes
    assert "segment_id" in stats.columns
    assert "mean_error" in stats.columns
    assert "size" in stats.columns
    assert stats["mean_error"].is_monotonic_decreasing


def test_dt_segmentation_get_decision_rules(sample_data):
    """Test extracting decision rules from the tree."""
    x_test = sample_data["x_test"]
    y_test = sample_data["y_test"]
    y_pred = sample_data["y_pred"]

    dt_seg = DecisionTreeSegmentation(max_depth=3)
    dt_seg.fit(x_test, y_test, y_pred, feature_names=["feature1", "feature2"])

    rules = dt_seg.get_decision_rules(n_segments=3)

    assert isinstance(rules, dict)
    assert len(rules) <= 3  # Might be fewer if there are fewer segments

    # Check structure of rules
    for segment_id, rule_list in rules.items():
        assert isinstance(segment_id, int)
        assert isinstance(rule_list, list)

        for rule in rule_list:
            assert "feature" in rule
            assert "operator" in rule
            assert "threshold" in rule
            assert rule["feature"] in ["feature1", "feature2"]
            assert rule["operator"] in ["≤", ">"]


def test_error_before_fit():
    """Test error handling when methods are called before fitting."""
    dt_seg = DecisionTreeSegmentation()

    # Methods that should raise errors if called before fit
    with pytest.raises(ValueError):
        dt_seg.get_segment_df()

    with pytest.raises(ValueError):
        dt_seg.get_segment_stats()

    with pytest.raises(ValueError):
        dt_seg.get_decision_rules()

    with pytest.raises(ValueError):
        dt_seg.get_feature_ranges()

    with pytest.raises(ValueError):
        plot_dt_segmentation(dt_seg)


def test_get_path_to_node(sample_data):
    """Test internal method for getting path to node."""
    x_test = sample_data["x_test"]
    y_test = sample_data["y_test"]
    y_pred = sample_data["y_pred"]

    dt_seg = DecisionTreeSegmentation(max_depth=2)  # Small tree for predictable testing
    dt_seg.fit(x_test, y_test, y_pred, feature_names=["feature1", "feature2"])

    # Get a leaf node id
    leaf_id = None
    for i in range(dt_seg.tree_model.tree_.node_count):
        if dt_seg.tree_model.tree_.children_left[i] == dt_seg.tree_model.tree_.children_right[i]:
            leaf_id = i
            break

    if leaf_id is not None:
        path = dt_seg._get_path_to_node(leaf_id)
        assert isinstance(path, list)
        assert path[0] == 0  # First node should be root (0)
        assert path[-1] == leaf_id  # Last node should be our target
