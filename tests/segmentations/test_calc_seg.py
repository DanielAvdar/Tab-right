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


def test_init_with_df():
    """Test initialization with DataFrame."""
    df = pd.DataFrame({"feature": [1, 2, 3, 4, 5], "label": [10, 20, 30, 40, 50], "pred": [11, 21, 28, 42, 49]})

    # Test with basic parameters
    seg_stats = SegmentationStats(
        df=df, label_col="label", feature="feature", prediction_col="pred", metric=mean_absolute_error
    )

    assert seg_stats.feature == "feature"
    assert seg_stats.label_col == "label"
    assert seg_stats.prediction_col == "pred"
    assert seg_stats.metric == mean_absolute_error
    assert not seg_stats.is_categorical


def test_init_with_backward_compatibility():
    """Test initialization with backward compatible parameter names."""
    df = pd.DataFrame({"feature": [1, 2, 3, 4, 5], "label": [10, 20, 30, 40, 50], "pred": [11, 21, 28, 42, 49]})

    # Test backward compatibility with pred_col parameter
    seg_stats = SegmentationStats(
        df=df, label_col="label", feature="feature", pred_col="pred", metric=mean_absolute_error
    )

    assert seg_stats.prediction_col == "pred"


def test_prepare_segments(sample_data):
    """Test segment preparation functionality."""
    # Test with continuous feature
    seg_stats = SegmentationStats(
        df=sample_data, label_col="y_true", feature="feature1", prediction_col="y_pred", metric=mean_absolute_error
    )

    segments = seg_stats._prepare_segments(sample_data, "feature1")
    assert len(segments) == len(sample_data)
    assert isinstance(segments, pd.Series)
    assert segments.nunique() <= 10  # Default bins=10

    # Test with categorical feature
    seg_stats = SegmentationStats(
        df=sample_data,
        label_col="y_true",
        feature="category",
        prediction_col="y_pred",
        metric=mean_absolute_error,
        is_categorical=True,
    )

    segments = seg_stats._prepare_segments(sample_data, "category")
    assert len(segments) == len(sample_data)
    assert segments.nunique() == 3  # A, B, C categories


def test_call_method_with_metric(sample_data):
    """Test calling the segmentation stats with a metric."""
    # Initialize with a metric
    seg_stats = SegmentationStats(
        df=sample_data, label_col="y_true", feature="feature1", prediction_col="y_pred", metric=mean_absolute_error
    )

    # Run with default metric
    result = seg_stats()
    assert isinstance(result, pd.DataFrame)
    assert "segment_id" in result.columns
    assert "score" in result.columns

    # Define custom metric and pass it to call
    def custom_metric(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred) ** 2)

    # Run with custom metric
    result = seg_stats(metric=custom_metric)
    assert isinstance(result, pd.DataFrame)
    assert "segment_id" in result.columns
    assert "score" in result.columns


def test_probability_mode(sample_data):
    """Test probability mode for multi-class classification."""
    # Initialize with list of probability columns
    seg_stats = SegmentationStats(
        df=sample_data,
        label_col=["prob_class1", "prob_class2"],
        feature="feature1",
        prediction_col="y_pred",  # Not used in probability mode, but required
    )

    # Run in probability mode
    result = seg_stats._run_probability_mode()
    assert isinstance(result, pd.DataFrame)
    assert "segment_id" in result.columns
    assert "score" in result.columns

    # Check that scores are dictionaries
    assert isinstance(result["score"].iloc[0], dict)
    assert "prob_class1" in result["score"].iloc[0]
    assert "prob_class2" in result["score"].iloc[0]


def test_metric_mode(sample_data):
    """Test metric mode for regression tasks."""

    def mae(y_true, y_pred):
        return np.abs(y_true - y_pred).mean()

    # Initialize for regression task
    seg_stats = SegmentationStats(
        df=sample_data, label_col="y_true", feature="feature1", prediction_col="y_pred", metric=mae
    )

    # Run in metric mode
    result = seg_stats._run_metric_mode()
    assert isinstance(result, pd.DataFrame)
    assert "segment_id" in result.columns
    assert "score" in result.columns
    assert result["score"].dtype == float


def test_check_method(sample_data):
    """Test the check method for validating data."""
    # Valid regression data
    seg_stats = SegmentationStats(
        df=sample_data, label_col="y_true", feature="feature1", prediction_col="y_pred", metric=mean_absolute_error
    )

    # Should not raise exceptions
    seg_stats.check()

    # Valid probability data
    seg_stats = SegmentationStats(
        df=sample_data, label_col=["prob_class1", "prob_class2"], feature="feature1", prediction_col="y_pred"
    )

    # Should not raise exceptions
    seg_stats.check()

    # Create invalid data with NaN
    bad_data = sample_data.copy()
    bad_data.loc[0, "y_true"] = np.nan

    seg_stats = SegmentationStats(
        df=bad_data, label_col="y_true", feature="feature1", prediction_col="y_pred", metric=mean_absolute_error
    )

    # Should raise exception
    with pytest.raises(ValueError, match="Label column contains NaN values"):
        seg_stats.check()

    # Create invalid probability data (doesn't sum to 1)
    bad_prob_data = sample_data.copy()
    bad_prob_data.loc[0, "prob_class1"] = 0.7
    bad_prob_data.loc[0, "prob_class2"] = 0.7  # Sum > 1

    seg_stats = SegmentationStats(
        df=bad_prob_data, label_col=["prob_class1", "prob_class2"], feature="feature1", prediction_col="y_pred"
    )

    # Should raise exception
    with pytest.raises(ValueError, match="Probabilities .* do not sum to 1"):
        seg_stats.check()


def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test missing required parameters
    with pytest.raises(ValueError, match="Either .*must be provided"):
        SegmentationStats()

    # Test missing metric when calling
    df = pd.DataFrame({"feature": [1, 2, 3], "label": [10, 20, 30], "pred": [11, 19, 32]})

    seg_stats = SegmentationStats(df=df, label_col="label", feature="feature", prediction_col="pred", metric=None)

    with pytest.raises(ValueError, match="Metric function must be provided"):
        seg_stats()
