"""Tests for the decision tree segmentation functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from tab_right.segmentations.find_seg import DecisionTreeSegmentation
from .test_utils import error_metric, create_decision_tree_model


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


@pytest.mark.parametrize("method_name", [
    "get_feature_ranges",
    "get_segment_df",
    "get_decision_rules",
])
def test_methods_raise_error_when_not_fitted(method_name):
    dt_seg = DecisionTreeSegmentation()
    with pytest.raises(ValueError, match="Model not fitted"):
        getattr(dt_seg, method_name)()


@pytest.mark.parametrize(
    "method_name, n_segments, expected_type, expected_columns_or_keys",
    [
        ("get_feature_ranges", None, list, None),
        ("get_segment_df", 3, pd.DataFrame, ["segment_id", "mean_error", "median_error", "max_error", "size", "feature1", "feature2"]),
        ("get_decision_rules", 3, dict, ["feature", "operator", "threshold"]),
    ]
)
def test_methods_on_fitted_model(training_data, method_name, n_segments, expected_type, expected_columns_or_keys):
    dt_seg = DecisionTreeSegmentation(max_depth=3)
    dt_seg.fit(
        training_data["x_test"],
        training_data["y_test"],
        training_data["y_pred"],
        feature_names=["feature1", "feature2"],
    )
    if n_segments is not None:
        result = getattr(dt_seg, method_name)(n_segments=n_segments)
    else:
        result = getattr(dt_seg, method_name)()
    assert isinstance(result, expected_type)
    if expected_columns_or_keys:
        if isinstance(result, pd.DataFrame):
            for col in expected_columns_or_keys:
                assert col in result.columns
        elif isinstance(result, dict):
            for rule_list in result.values():
                for rule in rule_list:
                    for key in expected_columns_or_keys:
                        assert key in rule


@pytest.mark.parametrize(
    "call_type, feature_col, max_depth",
    [
        ("original", None, 2),
        ("protocol", "feature1", 1),
        ("integration", "feature1", 2),
    ]
)
def test_groupby_output(sample_data, call_type, feature_col, max_depth):
    if call_type == "original":
        dt_seg = DecisionTreeSegmentation(
            df=sample_data.copy(),
            label_col="y_true",
            prediction_col="y_pred",
            feature1_col="feature1",
            feature2_col="feature2",
        )
        dt_seg.df["abs_error"] = abs(dt_seg.df["y_true"] - dt_seg.df["y_pred"])
        dt_seg.error_col = "abs_error"
        from sklearn.tree import DecisionTreeRegressor
        dt_seg.tree_model = DecisionTreeRegressor(max_depth=max_depth).fit(
            dt_seg.df[["feature1", "feature2"]], dt_seg.df["abs_error"])
        grouped = dt_seg()
    else:
        dt_seg = DecisionTreeSegmentation(
            df=sample_data.copy(),
            label_col="y_true",
            prediction_col="y_pred",
            feature1_col="feature1",
            feature2_col="feature2",
        )
        grouped = dt_seg(
            feature_col=feature_col,
            error_metric=error_metric,
            model=create_decision_tree_model(max_depth=max_depth)
        )
    assert hasattr(grouped, "groups")
    assert isinstance(grouped, pd.core.groupby.DataFrameGroupBy)
    if call_type != "original":
        assert dt_seg.tree_model is not None
