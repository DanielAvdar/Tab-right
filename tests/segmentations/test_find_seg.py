"""Tests for the decision tree segmentation functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from tab_right.segmentations.find_seg import DecisionTreeSegmentation


@pytest.fixture
def sample_data():
    """Create sample data for testing decision tree segmentation."""
    np.random.seed(42)
    n_samples = 100

    # Create a DataFrame with features and targets
    df = pd.DataFrame({
        "feature1": np.random.uniform(-1, 1, n_samples),
        "feature2": np.random.uniform(0, 10, n_samples),
        "feature3": np.random.normal(0, 1, n_samples),
    })

    # Create target variable with some dependency on features
    df["y_true"] = 2 * df["feature1"] ** 2 - df["feature2"] * 0.5 + np.random.normal(0, 1, n_samples)

    # Create predictions with some errors
    df["y_pred"] = df["y_true"] + 0.1 * df["feature1"] + 0.05 * df["feature2"] + np.random.normal(0, 0.5, n_samples)

    return df


@pytest.fixture
def training_data():
    """Create training and test data for model fitting."""
    np.random.seed(42)
    n_samples = 200
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


def test_init():
    """Test initialization of DecisionTreeSegmentation class."""
    # Test default initialization
    dt_seg = DecisionTreeSegmentation()
    assert dt_seg.max_depth == 5
    assert dt_seg.min_samples_leaf == 20
    assert dt_seg.df is None
    assert dt_seg.tree_model is None

    # Test with custom parameters
    dt_seg = DecisionTreeSegmentation(max_depth=3, min_samples_leaf=10)
    assert dt_seg.max_depth == 3
    assert dt_seg.min_samples_leaf == 10

    # Test with dataframe
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "y_true": [10, 20, 30], "y_pred": [11, 19, 31]})
    dt_seg = DecisionTreeSegmentation(
        df=df, label_col="y_true", prediction_col="y_pred", feature1_col="feature1", feature2_col="feature2"
    )
    assert dt_seg.df is not None
    assert dt_seg.label_col == "y_true"
    assert dt_seg.prediction_col == "y_pred"
    assert dt_seg.feature1_col == "feature1"
    assert dt_seg.feature2_col == "feature2"


def test_backward_compatibility_properties():
    """Test backward compatibility property accessors."""
    dt_seg = DecisionTreeSegmentation()

    # Test setting via old properties
    dt_seg.feature_1_col = "old_feature1"
    dt_seg.feature_2_col = "old_feature2"

    # Check that new properties were updated
    assert dt_seg.feature1_col == "old_feature1"
    assert dt_seg.feature2_col == "old_feature2"

    # Test getting via old properties
    assert dt_seg.feature_1_col == "old_feature1"
    assert dt_seg.feature_2_col == "old_feature2"

    # Test setting via new properties
    dt_seg.feature1_col = "new_feature1"
    dt_seg.feature2_col = "new_feature2"

    # Check that old properties reflect changes
    assert dt_seg.feature_1_col == "new_feature1"
    assert dt_seg.feature_2_col == "new_feature2"


def test_fit(training_data):
    """Test fitting the decision tree segmentation model."""
    x_test = training_data["x_test"]
    y_test = training_data["y_test"]
    y_pred = training_data["y_pred"]

    # Test fitting with specific feature names
    dt_seg = DecisionTreeSegmentation(max_depth=3)
    dt_seg.fit(x_test, y_test, y_pred, feature_names=["feature1", "feature2"])

    # Check if model was fitted correctly
    assert dt_seg.tree_model is not None
    assert dt_seg.feature1_col == "feature1"
    assert dt_seg.feature2_col == "feature2"
    assert dt_seg.error_col == "abs_error"
    assert dt_seg.df is not None
    assert dt_seg.label_col == "y_true"
    assert dt_seg.prediction_col == "y_pred"

    # Check that DataFrame has expected columns
    assert "feature1" in dt_seg.df.columns
    assert "feature2" in dt_seg.df.columns
    assert "abs_error" in dt_seg.df.columns
    assert "y_true" in dt_seg.df.columns
    assert "y_pred" in dt_seg.df.columns


def test_fit_numpy_array():
    """Test fitting with numpy arrays."""
    np.random.seed(42)
    # Create numpy arrays
    x = np.random.rand(100, 3)
    y_true = np.sin(x[:, 0]) + x[:, 1] ** 2
    y_pred = y_true + np.random.normal(0, 0.2, size=100)

    dt_seg = DecisionTreeSegmentation()
    dt_seg.fit(x, y_true, y_pred)

    # Check default feature names
    assert dt_seg.feature1_col == "Feature 0"
    assert dt_seg.feature2_col == "Feature 1"

    # Check with custom feature names
    dt_seg.fit(x, y_true, y_pred, feature_names=["X", "Y", "Z"])
    assert dt_seg.feature1_col == "X"
    assert dt_seg.feature2_col == "Y"


def test_train_tree_model(sample_data):
    """Test the train_tree_model method."""
    dt_seg = DecisionTreeSegmentation(
        df=sample_data, feature1_col="feature1", feature2_col="feature2", error_col="error"
    )

    # Add error column if it doesn't exist
    if "error" not in sample_data.columns:
        dt_seg.df["error"] = np.abs(sample_data["y_true"] - sample_data["y_pred"])

    # Train a new model
    model = DecisionTreeRegressor(max_depth=2)
    trained_model = dt_seg.train_tree_model(model)

    # Check that model was trained
    assert trained_model is not None
    assert hasattr(trained_model, "tree_")
    assert hasattr(trained_model, "predict")

    # Train with missing data should raise error
    dt_seg_empty = DecisionTreeSegmentation()
    with pytest.raises(ValueError, match="DataFrame and column names must be set before training"):
        dt_seg_empty.train_tree_model(DecisionTreeRegressor())


def test_calc_error():
    """Test the _calc_error classmethod."""

    # Define a test metric
    def abs_error(y_true, y_pred):
        return np.abs(y_true - y_pred.iloc[:, 0])

    # Create sample data
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.DataFrame({"pred": [1.1, 2.2, 2.9, 4.1, 5.5]})

    # Calculate errors
    errors = DecisionTreeSegmentation._calc_error(abs_error, y_true, y_pred)

    # Check results
    assert isinstance(errors, pd.Series)
    assert len(errors) == 5
    assert np.allclose(errors.values, np.array([0.1, 0.2, 0.1, 0.1, 0.5]), atol=1e-6)


def test_fit_model():
    """Test the _fit_model classmethod."""
    # Create sample data
    feature = pd.Series([1, 2, 3, 4, 5])
    error = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

    # Create model
    model = DecisionTreeRegressor(max_depth=2)

    # Fit model
    fitted_model = DecisionTreeSegmentation._fit_model(model, feature, error)

    # Check results
    assert fitted_model is not None
    assert hasattr(fitted_model, "tree_")

    # Test prediction
    pred = fitted_model.predict([[3]])
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1


def test_extract_leaves():
    """Test the _extract_leaves classmethod."""
    # Create and fit a simple model
    x = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]).reshape(-1, 1)
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])

    model = DecisionTreeRegressor(max_depth=1)
    model.fit(x, y)

    # Extract leaves
    leaves = DecisionTreeSegmentation._extract_leaves(model)

    # Check results
    assert isinstance(leaves, pd.DataFrame)
    assert "segment_id" in leaves.columns
    assert "segment_name" in leaves.columns
    assert len(leaves) == 2  # With max_depth=1, we should have 2 leaves


def test_call_method(sample_data):
    """Test the __call__ method."""
    # Create a segmentation instance
    dt_seg = DecisionTreeSegmentation(
        df=sample_data.copy(),
        label_col="y_true",
        prediction_col="y_pred",
        feature1_col="feature1",
        feature2_col="feature2",
    )

    # Add error column
    dt_seg.df["abs_error"] = np.abs(dt_seg.df["y_true"] - dt_seg.df["y_pred"])
    dt_seg.error_col = "abs_error"

    # Create and fit a model
    model = DecisionTreeRegressor(max_depth=2)
    dt_seg.tree_model = model.fit(dt_seg.df[["feature1", "feature2"]], dt_seg.df["abs_error"])

    # Call the model (original behavior)
    grouped = dt_seg()

    # Check results
    assert hasattr(grouped, "groups")
    assert isinstance(grouped, pd.core.groupby.DataFrameGroupBy)

    # Test protocol-compatible usage
    def error_metric(y_true, y_pred):
        return np.abs(y_true - y_pred.iloc[:, 0])

    # Call with protocol-compatible parameters
    grouped2 = dt_seg(feature_col="feature1", error_metric=error_metric, model=DecisionTreeRegressor(max_depth=1))

    # Check results of protocol usage
    assert hasattr(grouped2, "groups")
    assert isinstance(grouped2, pd.core.groupby.DataFrameGroupBy)


def test_get_feature_ranges(training_data):
    """Test getting feature ranges."""
    # Fit a model
    dt_seg = DecisionTreeSegmentation(max_depth=3)
    dt_seg.fit(
        training_data["x_test"],
        training_data["y_test"],
        training_data["y_pred"],
        feature_names=["feature1", "feature2"],
    )

    # Get feature ranges
    ranges = dt_seg.get_feature_ranges()

    # Check results
    assert isinstance(ranges, list)
    assert len(ranges) == 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in ranges)

    # For each range, the max should be >= min (allowing for equal values in edge cases)
    assert ranges[0][1] >= ranges[0][0]  # max >= min for feature 1
    assert ranges[1][1] >= ranges[1][0]  # max >= min for feature 2

    # Test error if tree not fitted
    dt_seg_empty = DecisionTreeSegmentation()
    with pytest.raises(ValueError, match="Model not fitted"):
        dt_seg_empty.get_feature_ranges()


def test_get_segment_df(training_data):
    """Test getting segment DataFrame."""
    # Fit a model
    dt_seg = DecisionTreeSegmentation(max_depth=3)
    dt_seg.fit(
        training_data["x_test"],
        training_data["y_test"],
        training_data["y_pred"],
        feature_names=["feature1", "feature2"],
    )

    # Get segment DataFrame
    segment_df = dt_seg.get_segment_df(n_segments=3)

    # Check results
    assert isinstance(segment_df, pd.DataFrame)
    assert "segment_id" in segment_df.columns
    assert "mean_error" in segment_df.columns
    assert "median_error" in segment_df.columns
    assert "max_error" in segment_df.columns
    assert "size" in segment_df.columns
    assert "feature1" in segment_df.columns
    assert "feature2" in segment_df.columns
    assert len(segment_df) <= 3  # Should have at most 3 segments

    # Test error if tree not fitted
    dt_seg_empty = DecisionTreeSegmentation()
    with pytest.raises(ValueError, match="Model not fitted"):
        dt_seg_empty.get_segment_df()


def test_get_decision_rules(training_data):
    """Test getting decision rules."""
    # Fit a model
    dt_seg = DecisionTreeSegmentation(max_depth=2)  # Small depth for predictable rules
    dt_seg.fit(
        training_data["x_test"],
        training_data["y_test"],
        training_data["y_pred"],
        feature_names=["feature1", "feature2"],
    )

    # Get decision rules
    rules = dt_seg.get_decision_rules(n_segments=3)

    # Check results
    assert isinstance(rules, dict)
    assert len(rules) <= 3  # Should have at most 3 segments

    # Check structure of rules
    for segment_id, rule_list in rules.items():
        assert isinstance(segment_id, int)
        assert isinstance(rule_list, list)

        for rule in rule_list:
            assert isinstance(rule, dict)
            assert "feature" in rule
            assert "operator" in rule
            assert "threshold" in rule
            assert rule["feature"] in ["feature1", "feature2"]
            assert rule["operator"] in ["≤", ">"]
            assert isinstance(rule["threshold"], float)

    # Test error if tree not fitted
    dt_seg_empty = DecisionTreeSegmentation()
    with pytest.raises(ValueError, match="Model not fitted"):
        dt_seg_empty.get_decision_rules()


def test_integration_with_protocols(sample_data):
    """Test integration with the FindSegmentation protocol."""
    # Create a segmentation instance
    dt_seg = DecisionTreeSegmentation(df=sample_data, label_col="y_true", prediction_col="y_pred")

    # Define an error metric as required by the protocol
    def error_metric(y_true, y_pred):
        if isinstance(y_pred, pd.DataFrame) and len(y_pred.columns) >= 1:
            return np.abs(y_true - y_pred.iloc[:, 0])
        return np.abs(y_true - y_pred)

    # Create a decision tree model
    model = DecisionTreeRegressor(max_depth=2)

    # Call as per the protocol definition
    result = dt_seg(feature_col="feature1", error_metric=error_metric, model=model)

    # Check results
    assert hasattr(result, "groups")
    assert isinstance(result, pd.core.groupby.DataFrameGroupBy)
