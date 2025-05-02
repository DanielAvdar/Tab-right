"""Tests for the double segmentation functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from tab_right.segmentations.double_seg import DecisionTreeDoubleSegmentation
from tab_right.segmentations.find_seg import DecisionTreeSegmentation


@pytest.fixture
def sample_data():
    """Create sample data for testing double segmentation."""
    np.random.seed(42)
    n_samples = 100

    # Create a DataFrame with features and targets
    df = pd.DataFrame({
        "feature1": np.random.uniform(-1, 1, n_samples),
        "feature2": np.random.uniform(0, 10, n_samples),
        "feature3": np.random.normal(0, 1, n_samples),
    })

    # Create target variable with some dependency on features
    noise = np.random.normal(0, 0.5, n_samples)
    df["y_true"] = df["feature1"] ** 2 + df["feature2"] * 0.5 + noise

    # Create predictions with systematic errors
    df["y_pred"] = (
        df["y_true"] + 0.1 * df["feature1"] + 0.05 * df["feature2"] ** 2 + np.random.normal(0, 0.3, n_samples)
    )

    return df


@pytest.fixture
def segmentation_finder(sample_data):
    """Create a segmentation finder instance for testing."""
    finder = DecisionTreeSegmentation(
        df=sample_data, label_col="y_true", prediction_col="y_pred", max_depth=3, min_samples_leaf=5
    )
    return finder


def test_init(segmentation_finder):
    """Test initialization of the double segmentation class."""
    double_seg = DecisionTreeDoubleSegmentation(segmentation_finder=segmentation_finder)
    assert double_seg.segmentation_finder == segmentation_finder


@pytest.mark.parametrize(
    "df1_data,df2_data,expected_rows,expected_columns",
    [
        (
            # First dataframe with 3 segments
            {"segment_id": [1, 2, 3], "segment_name": ["Low", "Medium", "High"], "score": [0.1, 0.2, 0.3]},
            # Second dataframe with 2 segments
            {"segment_id": [10, 20], "segment_name": ["Young", "Old"], "score": [0.15, 0.25]},
            # Expected number of rows and columns to check
            6,  # 3 x 2 = 6 rows from cross join
            ["feature_1", "feature_2", "score", "segment_id"],  # Expected columns
        ),
    ],
)
def test_combine_2_features(df1_data, df2_data, expected_rows, expected_columns):
    """Test the combine_2_features method with parametrized inputs."""
    # Create sample DataFrames to combine
    df1 = pd.DataFrame(df1_data)
    df2 = pd.DataFrame(df2_data)

    # Combine the DataFrames
    result = DecisionTreeDoubleSegmentation._combine_2_features(df1, df2)

    # Check the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected_rows  # Cross join should multiply the rows
    
    # Check that expected columns exist
    for column in expected_columns:
        assert column in result.columns
    
    # Check the values for first dataframe's segment names
    for segment_name in df1_data["segment_name"]:
        assert segment_name in result["feature_1"].values
    
    # Check the values for second dataframe's segment names
    for segment_name in df2_data["segment_name"]:
        assert segment_name in result["feature_2"].values


@pytest.mark.parametrize(
    "segments_data,expected_groups",
    [
        (
            # Sample segments with 3 distinct values
            [1, 1, 2, 2, 3],
            3,  # Expected number of groups
        ),
    ],
)
def test_group_by_segment(sample_data, segments_data, expected_groups):
    """Test the group_by_segment method with parametrized inputs."""
    # Create a sample segment Series
    segments = pd.Series(segments_data, name="segment")

    # Group the data
    grouped = DecisionTreeDoubleSegmentation._group_by_segment(sample_data.iloc[:5], segments)

    # Check the result
    assert hasattr(grouped, "groups")
    assert len(grouped.groups) == expected_groups  # Should have expected number of groups
    
    # Check that all segment values are present as groups
    for segment_value in set(segments_data):
        assert segment_value in grouped.groups


@pytest.mark.parametrize(
    "feature1_col,feature2_col,model_depth,expected_columns",
    [
        (
            "feature1",
            "feature2",
            2,  # Smaller depth for predictable results
            ["segment_id", "feature_1", "feature_2", "score"],  # Expected columns
        ),
    ],
)
def test_call_method(segmentation_finder, sample_data, feature1_col, feature2_col, model_depth, expected_columns):
    """Test the __call__ method with parametrized inputs."""
    # Create a double segmentation instance
    double_seg = DecisionTreeDoubleSegmentation(segmentation_finder=segmentation_finder)

    # Define a simple error metric
    def error_metric(y_true, y_pred):
        return np.abs(y_true - y_pred.iloc[:, 0])

    # Create a decision tree model
    model = DecisionTreeRegressor(max_depth=model_depth)

    # Call the double segmentation
    result = double_seg(feature1_col=feature1_col, feature2_col=feature2_col, error_metric=error_metric, model=model)

    # Check the result
    assert isinstance(result, pd.DataFrame)
    
    # Check that expected columns exist
    for column in expected_columns:
        assert column in result.columns

    # Verify the segmentation worked correctly
    assert len(result) > 0
    assert result["score"].notna().all()


@pytest.mark.parametrize(
    "depth,min_segments",
    [
        (2, 4),  # Depth 2 should give at least 2x2=4 segments
    ],
)
def test_end_to_end(sample_data, depth, min_segments):
    """Test the entire double segmentation workflow with parametrized inputs."""
    # Create a finder with a pre-fitted tree model
    finder = DecisionTreeSegmentation(df=sample_data, label_col="y_true", prediction_col="y_pred", max_depth=depth)

    # Create the double segmentation instance
    double_seg = DecisionTreeDoubleSegmentation(segmentation_finder=finder)

    # Define an error metric
    def error_metric(y_true, y_pred):
        if isinstance(y_pred, pd.DataFrame) and len(y_pred.columns) >= 1:
            pred_values = y_pred.iloc[:, 0]
        else:
            pred_values = y_pred
        return np.abs(y_true - pred_values)

    # Create a decision tree model
    model = DecisionTreeRegressor(max_depth=depth)

    # Run the segmentation
    result = double_seg(feature1_col="feature1", feature2_col="feature2", error_metric=error_metric, model=model)

    # Check we have meaningful segmentation
    assert len(result) >= min_segments  # Should have at least min_segments

    # Check segments are named correctly
    assert "feature_1" in result.columns
    assert "feature_2" in result.columns

    # Check scores are valid
    assert result["score"].notna().all()
    assert (result["score"] >= 0).all()

    # Different feature combinations should have different scores
    assert result["score"].nunique() > 1
