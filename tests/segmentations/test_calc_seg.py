"""Tests for the segmentation statistics functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from tab_right.segmentations.calc_seg import SegmentationStats
from tests.test_utils import error_metric, create_decision_tree_model


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
    "mode,label_col,expected_score_type,metric_fn",
    [
        ("probability", ["prob_class1", "prob_class2"], dict, None),  # Test probability mode
        ("metric", "y_true", float, lambda y_true, y_pred: float(np.mean(np.abs(y_true - y_pred)))),  # Test metric mode with scalar output
    ]
)
def test_run_modes(sample_data, mode, label_col, expected_score_type, metric_fn):
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
            metric=metric_fn
        )
        result = seg_stats._run_metric_mode()
    assert isinstance(result, pd.DataFrame)
    assert "segment_id" in result.columns
    assert "score" in result.columns
    assert isinstance(result["score"].iloc[0], expected_score_type)


# --- Start Refactored Validation Tests ---

@pytest.mark.parametrize(
    "setup_func,expected_exception,expected_message",
    [
        # Missing essential parameters
        (lambda sample_data: SegmentationStats(), ValueError, "Either .*must be provided"),
        # Label column contains NaN
        (lambda sample_data: SegmentationStats(
            df=sample_data.assign(y_true=np.where(sample_data.index == 0, np.nan, sample_data["y_true"])),
            label_col="y_true", feature="feature1", prediction_col="y_pred", metric=mean_absolute_error),
         ValueError, "Label column contains NaN values"),
        # Probabilities don't sum to 1
        (lambda sample_data: SegmentationStats(
            df=sample_data.assign(prob_class1=np.where(sample_data.index == 0, 0.7, sample_data["prob_class1"]),
                                    prob_class2=np.where(sample_data.index == 0, 0.7, sample_data["prob_class2"])),
            label_col=["prob_class1", "prob_class2"], feature="feature1", prediction_col="y_pred"),
         ValueError, "Probabilities .* do not sum to 1"),
        # Metric missing for metric mode
        (lambda sample_data: SegmentationStats(
            df=pd.DataFrame({"feature": [1, 2, 3], "label": [10, 20, 30], "pred": [11, 19, 32]}),
            label_col="label", feature="feature", prediction_col="pred", metric=None),
         ValueError, "Metric function must be provided"),
    ]
)
def test_validation_errors(sample_data, setup_func, expected_exception, expected_message):
    """Test validation error cases in a parameterized way."""
    if "Metric function must be provided" in expected_message:
        with pytest.raises(expected_exception, match=expected_message):
            setup_func(sample_data)()
    else:
        with pytest.raises(expected_exception, match=expected_message):
            setup_func(sample_data).check()

# --- End Refactored Validation Tests ---
