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


def test_backward_compatibility_property_accessors():
    """Test backward compatibility property accessors for feature column names."""
    # Create a new instance
    dt_seg = DecisionTreeSegmentation()

    # Test the setters (these should update the new attribute names)
    dt_seg.feature_1_col = "old_feature1"
    dt_seg.feature_2_col = "old_feature2"

    # Verify that the new attribute names were updated
    assert dt_seg.feature1_col == "old_feature1"
    assert dt_seg.feature2_col == "old_feature2"

    # Test the getters (these should access the new attribute names)
    assert dt_seg.feature_1_col == "old_feature1"
    assert dt_seg.feature_2_col == "old_feature2"

    # Update via the new attribute names
    dt_seg.feature1_col = "new_feature1"
    dt_seg.feature2_col = "new_feature2"

    # Verify that the old property accessors reflect the changes
    assert dt_seg.feature_1_col == "new_feature1"
    assert dt_seg.feature_2_col == "new_feature2"


def test_train_tree_model_error_handling():
    """Test error handling in train_tree_model method."""
    dt_seg = DecisionTreeSegmentation()

    # Should raise ValueError when DataFrame is None
    with pytest.raises(ValueError, match="DataFrame and column names must be set before training"):
        dt_seg.train_tree_model(RandomForestRegressor())

    # Set DataFrame but keep columns None
    dt_seg.df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4], "error": [0.1, 0.2]})

    # Should still raise ValueError when column names are None
    with pytest.raises(ValueError, match="DataFrame and column names must be set before training"):
        dt_seg.train_tree_model(RandomForestRegressor())


def test_call_method_error_handling():
    """Test error handling in the __call__ method."""
    dt_seg = DecisionTreeSegmentation()

    # Should raise ValueError when DataFrame is None
    with pytest.raises(ValueError, match="DataFrame must be set before calling"):
        dt_seg(RandomForestRegressor())

    # Set DataFrame but keep tree_model None
    dt_seg.df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "error": [0.1, 0.2]})
    dt_seg.feature1_col = "feature1"
    dt_seg.feature2_col = "feature2"
    dt_seg.error_col = "error"

    # Should raise ValueError when model is None and not provided
    with pytest.raises(ValueError, match="Model not fitted"):
        dt_seg(None)


def test_old_attribute_names_backward_compatibility():
    """Test backward compatibility with old attribute names (feature_1_col, feature_2_col)."""
    # Create simple test data
    dt_seg = DecisionTreeSegmentation()

    # Test the old attribute names (this should target lines 198-206)
    dt_seg.feature_1_col = "old_feature1"
    dt_seg.feature_2_col = "old_feature2"

    # Verify the properties work in both directions
    assert dt_seg.feature1_col == "old_feature1"
    assert dt_seg.feature2_col == "old_feature2"
    assert dt_seg.feature_1_col == "old_feature1"
    assert dt_seg.feature_2_col == "old_feature2"

    # Test that setting via new attribute names works too
    dt_seg.feature1_col = "new_feature1"
    dt_seg.feature2_col = "new_feature2"

    # Verify the old property getter reflects the changes
    assert dt_seg.feature_1_col == "new_feature1"
    assert dt_seg.feature_2_col == "new_feature2"


def test_edge_case_feature_ranges():
    """Test the edge case handling in get_feature_ranges method."""
    # Create a mock DecisionTreeSegmentation with a tree that doesn't have any splits on X or Y
    dt_seg = DecisionTreeSegmentation()

    # Create minimal test data
    x = np.array([[1, 2], [3, 4]])
    y_true = np.array([1, 2])
    y_pred = np.array([1.1, 1.9])

    # Fit model with only 2 data points and max_depth=1 to create a very simple tree
    dt_seg.fit(x, y_true, y_pred, feature_names=["feature1", "feature2"])

    # Now replace the trained tree model with a mock that will trigger the edge cases
    # This tests lines 459 and 461 where feature ranges can't be determined
    original_tree = dt_seg.tree_model

    # Create a class to simulate a tree without splits on certain features
    class MockTreeModel:
        def __init__(self):
            self.tree_ = original_tree.tree_

        def predict(self, x_input):
            return np.ones(len(x_input))

        def apply(self, x_input):
            return np.zeros(len(x_input), dtype=int)

    # Patch the tree feature/threshold arrays to simulate a tree with no splits
    # on one of the features (this will test the edge case in get_feature_ranges)
    dt_seg.tree_model = MockTreeModel()

    # Get feature ranges - this would trigger default values (-1, 1) for features without splits
    ranges = dt_seg.get_feature_ranges()

    # Verify we got ranges even without proper splits in the tree
    assert len(ranges) == 2
    assert all(len(range_pair) == 2 for range_pair in ranges)
    # Check that padding was applied correctly
    assert all(r[1] > r[0] for r in ranges)
