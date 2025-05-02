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


@pytest.mark.parametrize(
    "params,expected",
    [
        # Test default initialization
        (
            {},
            {"max_depth": 5, "min_samples_leaf": 20, "df": None, "tree_model": None},
        ),
        # Test with custom parameters
        (
            {"max_depth": 3, "min_samples_leaf": 10},
            {"max_depth": 3, "min_samples_leaf": 10},
        ),
        # Test with dataframe
        (
            {
                "df": pd.DataFrame({
                    "feature1": [1, 2, 3], "feature2": [4, 5, 6], 
                    "y_true": [10, 20, 30], "y_pred": [11, 19, 31]
                }),
                "label_col": "y_true",
                "prediction_col": "y_pred",
                "feature1_col": "feature1",
                "feature2_col": "feature2",
            },
            {
                "label_col": "y_true", 
                "prediction_col": "y_pred", 
                "feature1_col": "feature1", 
                "feature2_col": "feature2"
            },
        ),
    ],
)
def test_init(params, expected):
    """Test initialization of DecisionTreeSegmentation class with parametrized inputs."""
    dt_seg = DecisionTreeSegmentation(**params)
    
    for attr, value in expected.items():
        if attr == "df" and value is None:
            assert dt_seg.df is None
        elif attr == "tree_model" and value is None:
            assert dt_seg.tree_model is None
        elif attr == "df" and value is not None:
            assert dt_seg.df is not None
        else:
            assert getattr(dt_seg, attr) == value


@pytest.mark.parametrize(
    "old_names,new_names",
    [
        (
            {"feature_1_col": "old_feature1", "feature_2_col": "old_feature2"},
            {"feature1_col": "new_feature1", "feature2_col": "new_feature2"},
        ),
    ],
)
def test_backward_compatibility_properties(old_names, new_names):
    """Test backward compatibility property accessors."""
    dt_seg = DecisionTreeSegmentation()

    # Test setting via old properties
    for old_prop, old_val in old_names.items():
        setattr(dt_seg, old_prop, old_val)

    # Check that new properties were updated
    assert dt_seg.feature1_col == old_names["feature_1_col"]
    assert dt_seg.feature2_col == old_names["feature_2_col"]

    # Test getting via old properties
    assert dt_seg.feature_1_col == old_names["feature_1_col"]
    assert dt_seg.feature_2_col == old_names["feature_2_col"]

    # Test setting via new properties
    for new_prop, new_val in new_names.items():
        setattr(dt_seg, new_prop, new_val)

    # Check that old properties reflect changes
    assert dt_seg.feature_1_col == new_names["feature1_col"]
    assert dt_seg.feature_2_col == new_names["feature2_col"]


@pytest.mark.parametrize(
    "max_depth,feature_names,expected_cols",
    [
        (
            3,
            ["feature1", "feature2"],
            ["feature1", "feature2", "abs_error", "y_true", "y_pred"],
        ),
    ],
)
def test_fit(training_data, max_depth, feature_names, expected_cols):
    """Test fitting the decision tree segmentation model."""
    x_test = training_data["x_test"]
    y_test = training_data["y_test"]
    y_pred = training_data["y_pred"]

    # Test fitting with specific feature names
    dt_seg = DecisionTreeSegmentation(max_depth=max_depth)
    dt_seg.fit(x_test, y_test, y_pred, feature_names=feature_names)

    # Check if model was fitted correctly
    assert dt_seg.tree_model is not None
    assert dt_seg.feature1_col == feature_names[0]
    assert dt_seg.feature2_col == feature_names[1]
    assert dt_seg.error_col == "abs_error"
    assert dt_seg.df is not None
    assert dt_seg.label_col == "y_true"
    assert dt_seg.prediction_col == "y_pred"

    # Check that DataFrame has expected columns
    for col in expected_cols:
        assert col in dt_seg.df.columns


@pytest.mark.parametrize(
    "feature_names,expected_feature_names",
    [
        (None, ["Feature 0", "Feature 1"]),  # Default feature names
        (["X", "Y", "Z"], ["X", "Y"]),  # Custom feature names
    ],
)
def test_fit_numpy_array(feature_names, expected_feature_names):
    """Test fitting with numpy arrays."""
    np.random.seed(42)
    # Create numpy arrays
    x = np.random.rand(100, 3)
    y_true = np.sin(x[:, 0]) + x[:, 1] ** 2
    y_pred = y_true + np.random.normal(0, 0.2, size=100)

    dt_seg = DecisionTreeSegmentation()
    dt_seg.fit(x, y_true, y_pred, feature_names=feature_names)

    # Check feature names
    assert dt_seg.feature1_col == expected_feature_names[0]
    assert dt_seg.feature2_col == expected_feature_names[1]


@pytest.mark.parametrize(
    "test_case,should_raise",
    [
        ("valid_model", False),
    ],
)
def test_train_tree_model_valid(sample_data, test_case, should_raise):
    """Test the train_tree_model method for valid cases."""
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


def test_train_tree_model_error_empty():
    """Test train_tree_model raises error when df/columns are not set."""
    dt_seg_empty = DecisionTreeSegmentation()
    with pytest.raises(ValueError, match="DataFrame and column names must be set before training"):
        dt_seg_empty.train_tree_model(DecisionTreeRegressor())


@pytest.mark.parametrize(
    "mode,feature_col,max_depth",
    [
        ("original", None, 2),  # Original behavior
    ],
)
def test_call_method_original(sample_data, mode, feature_col, max_depth):
    """Test the __call__ method (original behavior)."""
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
    model = DecisionTreeRegressor(max_depth=max_depth) # Use max_depth from params
    dt_seg.tree_model = model.fit(dt_seg.df[["feature1", "feature2"]], dt_seg.df["abs_error"])

    # Call the model (original behavior)
    grouped = dt_seg()
    
    # Check results
    assert hasattr(grouped, "groups")
    assert isinstance(grouped, pd.core.groupby.DataFrameGroupBy)


@pytest.mark.parametrize(
    "feature_col,max_depth",
    [
        ("feature1", 1),  # Protocol-compatible behavior
    ],
)
def test_call_method_protocol(sample_data, feature_col, max_depth):
    """Test the __call__ method (protocol-compatible behavior)."""
    # Create a segmentation instance
    dt_seg = DecisionTreeSegmentation(
        df=sample_data.copy(),
        label_col="y_true",
        prediction_col="y_pred",
        feature1_col="feature1", # Set features for context, though protocol uses feature_col
        feature2_col="feature2",
    )

    # Define a simple error metric for protocol usage
    def error_metric(y_true, y_pred):
        return np.abs(y_true - y_pred.iloc[:, 0])

    # Call with protocol-compatible parameters
    # The model is trained internally when called this way
    grouped = dt_seg(feature_col=feature_col, error_metric=error_metric, model=DecisionTreeRegressor(max_depth=max_depth))
    
    # Check results
    assert hasattr(grouped, "groups")
    assert isinstance(grouped, pd.core.groupby.DataFrameGroupBy)
    assert dt_seg.tree_model is not None # Check model was trained internally


@pytest.mark.parametrize(
    "test_case,should_raise,max_depth,expected_ranges",
    [
        ("valid_model", False, 3, 2),  # Valid model, 2 feature ranges
    ],
)
def test_get_feature_ranges_valid(training_data, test_case, should_raise, max_depth, expected_ranges):
    """Test getting feature ranges for a valid fitted model."""
    # Fit a model
    dt_seg = DecisionTreeSegmentation(max_depth=max_depth)
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
    assert len(ranges) == expected_ranges
    assert all(isinstance(r, tuple) and len(r) == 2 for r in ranges)

    # For each range, the max should be >= min (allowing for equal values in edge cases)
    assert ranges[0][1] >= ranges[0][0]  # max >= min for feature 1
    assert ranges[1][1] >= ranges[1][0]  # max >= min for feature 2


def test_get_feature_ranges_error_empty():
    """Test get_feature_ranges raises error when model is not fitted."""
    dt_seg_empty = DecisionTreeSegmentation()
    with pytest.raises(ValueError, match="Model not fitted"):
        dt_seg_empty.get_feature_ranges()


@pytest.mark.parametrize(
    "test_case,should_raise,max_depth,n_segments,expected_columns",
    [
        (
            "valid_model", 
            False, 
            3, 
            3,  # Number of segments to return
            ["segment_id", "mean_error", "median_error", "max_error", "size", "feature1", "feature2"]  # Expected columns
        ),
    ],
)
def test_get_segment_df_valid(training_data, test_case, should_raise, max_depth, n_segments, expected_columns):
    """Test getting segment DataFrame for a valid fitted model."""
    # Fit a model
    dt_seg = DecisionTreeSegmentation(max_depth=max_depth)
    dt_seg.fit(
        training_data["x_test"],
        training_data["y_test"],
        training_data["y_pred"],
        feature_names=["feature1", "feature2"],
    )

    # Get segment DataFrame
    segment_df = dt_seg.get_segment_df(n_segments=n_segments)

    # Check results
    assert isinstance(segment_df, pd.DataFrame)
    
    # Check expected columns
    for col in expected_columns:
        assert col in segment_df.columns
        
    assert len(segment_df) <= n_segments  # Should have at most n_segments


def test_get_segment_df_error_empty():
    """Test get_segment_df raises error when model is not fitted."""
    dt_seg_empty = DecisionTreeSegmentation()
    with pytest.raises(ValueError, match="Model not fitted"):
        dt_seg_empty.get_segment_df()


@pytest.mark.parametrize(
    "test_case,should_raise,max_depth,n_segments,expected_rule_keys",
    [
        (
            "valid_model", 
            False, 
            2,  # Small depth for predictable rules
            3,  # Number of segments to return
            ["feature", "operator", "threshold"]  # Expected keys in rule dictionaries
        ),
    ],
)
def test_get_decision_rules_valid(training_data, test_case, should_raise, max_depth, n_segments, expected_rule_keys):
    """Test getting decision rules for a valid fitted model."""
    # Fit a model
    dt_seg = DecisionTreeSegmentation(max_depth=max_depth)
    dt_seg.fit(
        training_data["x_test"],
        training_data["y_test"],
        training_data["y_pred"],
        feature_names=["feature1", "feature2"],
    )

    # Get decision rules
    rules = dt_seg.get_decision_rules(n_segments=n_segments)

    # Check results
    assert isinstance(rules, dict)
    assert len(rules) <= n_segments  # Should have at most n_segments

    # Check structure of rules
    for segment_id, rule_list in rules.items():
        assert isinstance(segment_id, int)
        assert isinstance(rule_list, list)

        for rule in rule_list:
            assert isinstance(rule, dict)
            # Check expected keys in each rule
            for key in expected_rule_keys:
                assert key in rule
            
            assert rule["feature"] in ["feature1", "feature2"]
            assert rule["operator"] in ["≤", ">"]
            assert isinstance(rule["threshold"], float)


def test_get_decision_rules_error_empty():
    """Test get_decision_rules raises error when model is not fitted."""
    dt_seg_empty = DecisionTreeSegmentation()
    with pytest.raises(ValueError, match="Model not fitted"):
        dt_seg_empty.get_decision_rules()


@pytest.mark.parametrize(
    "feature_col,max_depth",
    [
        ("feature1", 2),  # Test with feature1 and depth 2
    ],
)
def test_integration_with_protocols(sample_data, feature_col, max_depth):
    """Test integration with the FindSegmentation protocol."""
    # Create a segmentation instance
    dt_seg = DecisionTreeSegmentation(df=sample_data, label_col="y_true", prediction_col="y_pred")

    # Define an error metric as required by the protocol
    def error_metric(y_true, y_pred):
        if isinstance(y_pred, pd.DataFrame) and len(y_pred.columns) >= 1:
            return np.abs(y_true - y_pred.iloc[:, 0])
        return np.abs(y_true - y_pred)

    # Create a decision tree model
    model = DecisionTreeRegressor(max_depth=max_depth)

    # Call as per the protocol definition
    result = dt_seg(feature_col=feature_col, error_metric=error_metric, model=model)

    # Check results
    assert hasattr(result, "groups")
    assert isinstance(result, pd.core.groupby.DataFrameGroupBy)
