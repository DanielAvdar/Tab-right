"""Tests for the segmentation statistics functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from tab_right.segmentations.calc_seg import SegmentationStats


@pytest.fixture
def sample_data():
    """Create sample data for testing segmentation statistics."""
    np.random.seed(42)
    n_samples = 100

    # Create a DataFrame with features and targets
    df = pd.DataFrame({
        "feature1": np.random.uniform(-1, 1, n_samples),
        "feature2": np.random.uniform(0, 10, n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples),
    })

    # Create target variable with some dependency on features
    df["y_true"] = 2 * df["feature1"] + 0.5 * df["feature2"] + np.random.normal(0, 1, n_samples)

    # Create predictions with some errors
    df["y_pred"] = df["y_true"] + np.random.normal(0, 0.5, n_samples)

    # For classification case, create probability columns
    df["prob_class1"] = np.abs(np.sin(df["feature1"]))
    df["prob_class2"] = 1 - df["prob_class1"]

    return df


@pytest.mark.parametrize(
    "params,expected",
    [
        # Test with basic parameters
        (
            {
                "label_col": "label",
                "feature": "feature",
                "prediction_col": "pred",
                "metric": mean_absolute_error,
            },
            {
                "feature": "feature",
                "label_col": "label",
                "prediction_col": "pred",
                "metric": mean_absolute_error,
                "is_categorical": False,
            },
        ),
        # Test with backward compatibility
        (
            {
                "label_col": "label",
                "feature": "feature",
                "pred_col": "pred",
                "metric": mean_absolute_error,
            },
            {"prediction_col": "pred"},
        ),
    ],
)
def test_init_and_compatibility(params, expected):
    """Test initialization with DataFrame and backward compatibility."""
    df = pd.DataFrame({"feature": [1, 2, 3, 4, 5], "label": [10, 20, 30, 40, 50], "pred": [11, 21, 28, 42, 49]})
    
    # Test with provided parameters
    seg_stats = SegmentationStats(df=df, **params)
    
    # Check expected values
    for key, value in expected.items():
        assert getattr(seg_stats, key) == value


@pytest.mark.parametrize(
    "feature,is_categorical,expected_nunique", 
    [
        ("feature1", False, 10),  # Test continuous feature with default bins=10
        ("category", True, 3),    # Test categorical feature with 3 categories
    ]
)
def test_prepare_segments(sample_data, feature, is_categorical, expected_nunique):
    """Test segment preparation functionality."""
    seg_stats = SegmentationStats(
        df=sample_data, 
        label_col="y_true", 
        feature=feature, 
        prediction_col="y_pred", 
        metric=mean_absolute_error,
        is_categorical=is_categorical
    )

    segments = seg_stats._prepare_segments(sample_data, feature)
    assert len(segments) == len(sample_data)
    assert isinstance(segments, pd.Series)
    
    # For categorical features, we expect exact nunique
    # For continuous features with binning, it should be <= expected
    if is_categorical:
        assert segments.nunique() == expected_nunique
    else:
        assert segments.nunique() <= expected_nunique


@pytest.mark.parametrize(
    "metric_param,custom_metric", 
    [
        (mean_absolute_error, None),  # Test with default metric
        (mean_absolute_error, lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred) ** 2)),  # Test with custom metric
    ]
)
def test_call_method_with_metric(sample_data, metric_param, custom_metric):
    """Test calling the segmentation stats with metrics."""
    # Initialize with a metric
    seg_stats = SegmentationStats(
        df=sample_data, 
        label_col="y_true", 
        feature="feature1", 
        prediction_col="y_pred", 
        metric=metric_param
    )

    # Run with provided metric or default
    if custom_metric:
        result = seg_stats(metric=custom_metric)
    else:
        result = seg_stats()
        
    assert isinstance(result, pd.DataFrame)
    assert "segment_id" in result.columns
    assert "score" in result.columns


@pytest.mark.parametrize(
    "mode,label_col,expected_score_type", 
    [
        ("probability", ["prob_class1", "prob_class2"], dict),  # Test probability mode
        ("metric", "y_true", float),  # Test metric mode
    ]
)
def test_run_modes(sample_data, mode, label_col, expected_score_type):
    """Test different run modes (probability and metric)."""
    # Initialize for appropriate mode
    if mode == "probability":
        seg_stats = SegmentationStats(
            df=sample_data,
            label_col=label_col,
            feature="feature1",
            prediction_col="y_pred",  # Not used in probability mode, but required
        )
        result = seg_stats._run_probability_mode()
        
        # Check that scores are dictionaries with expected keys
        assert all(key in result["score"].iloc[0] for key in label_col)
    else:
        seg_stats = SegmentationStats(
            df=sample_data, 
            label_col=label_col, 
            feature="feature1", 
            prediction_col="y_pred", 
            metric=lambda y_true, y_pred: np.abs(y_true - y_pred).mean()
        )
        result = seg_stats._run_metric_mode()
        
    assert isinstance(result, pd.DataFrame)
    assert "segment_id" in result.columns
    assert "score" in result.columns
    assert isinstance(result["score"].iloc[0], expected_score_type)


@pytest.mark.parametrize(
    "test_case,should_raise,error_type,error_match", 
    [
        # Valid cases
        ("valid_regression", False, None, None),
        ("valid_probability", False, None, None),
        # Invalid cases
        ("missing_parameters", True, ValueError, "Either .*must be provided"),
        ("nan_values", True, ValueError, "Label column contains NaN values"),
        ("invalid_probabilities", True, ValueError, "Probabilities .* do not sum to 1"),
        ("missing_metric", True, ValueError, "Metric function must be provided"),
    ]
)
def test_validation_and_errors(sample_data, test_case, should_raise, error_type, error_match):
    """Test validation and error handling."""
    if test_case == "valid_regression":
        seg_stats = SegmentationStats(
            df=sample_data, 
            label_col="y_true", 
            feature="feature1", 
            prediction_col="y_pred", 
            metric=mean_absolute_error
        )
        seg_stats.check()  # Should not raise
        
    elif test_case == "valid_probability":
        seg_stats = SegmentationStats(
            df=sample_data, 
            label_col=["prob_class1", "prob_class2"], 
            feature="feature1", 
            prediction_col="y_pred"
        )
        seg_stats.check()  # Should not raise
        
    elif test_case == "missing_parameters":
        with pytest.raises(error_type, match=error_match):
            SegmentationStats()
            
    elif test_case == "nan_values":
        bad_data = sample_data.copy()
        bad_data.loc[0, "y_true"] = np.nan
        seg_stats = SegmentationStats(
            df=bad_data, 
            label_col="y_true", 
            feature="feature1", 
            prediction_col="y_pred", 
            metric=mean_absolute_error
        )
        with pytest.raises(error_type, match=error_match):
            seg_stats.check()
            
    elif test_case == "invalid_probabilities":
        bad_prob_data = sample_data.copy()
        bad_prob_data.loc[0, "prob_class1"] = 0.7
        bad_prob_data.loc[0, "prob_class2"] = 0.7  # Sum > 1
        seg_stats = SegmentationStats(
            df=bad_prob_data, 
            label_col=["prob_class1", "prob_class2"], 
            feature="feature1", 
            prediction_col="y_pred"
        )
        with pytest.raises(error_type, match=error_match):
            seg_stats.check()
            
    elif test_case == "missing_metric":
        df = pd.DataFrame({"feature": [1, 2, 3], "label": [10, 20, 30], "pred": [11, 19, 32]})
        seg_stats = SegmentationStats(df=df, label_col="label", feature="feature", prediction_col="pred", metric=None)
        with pytest.raises(error_type, match=error_match):
            seg_stats()
