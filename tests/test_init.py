import pandas as pd
import pytest

from tab_right.segmentations.categorical import CategoricalSegmentationStats
from tab_right.segmentations.continuous import ContinuousSegmentationStats


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({
            "feature": ["a", "a", "b", "b"],
            "class_0": [0.7, 0.6, 0.2, 0.1],
            "class_1": [0.3, 0.4, 0.8, 0.9],
        }),
        pd.DataFrame({
            "feature": ["x", "x", "y", "y"],
            "class_0": [0.5, 0.5, 0.3, 0.3],
            "class_1": [0.5, 0.5, 0.7, 0.7],
        }),
    ],
)
def test_probability_mode_basic(df):
    seg = CategoricalSegmentationStats(df, label_col=["class_0", "class_1"], pred_col=None, feature="feature")
    result = seg.run()
    assert set(result["segment"]) == set(df["feature"].unique())
    for score in result["score"]:
        assert isinstance(score, dict)
        assert abs(sum(score.values()) - 1) < 1e-6


# Test check: NaN in probability columns
def test_check_nan_prob():
    df = pd.DataFrame({
        "feature": ["a", "b"],
        "class_0": [0.5, None],
        "class_1": [0.5, 1.0],
    })
    seg = CategoricalSegmentationStats(df, label_col=["class_0", "class_1"], pred_col=None, feature="feature")
    with pytest.raises(ValueError, match="NaN"):
        seg.check()


# Test check: probabilities do not sum to 1
def test_check_prob_sum():
    df = pd.DataFrame({
        "feature": ["a", "b"],
        "class_0": [0.6, 0.2],
        "class_1": [0.3, 0.7],
    })
    seg = CategoricalSegmentationStats(df, label_col=["class_0", "class_1"], pred_col=None, feature="feature")
    with pytest.raises(ValueError, match="sum to 1"):
        seg.check()


# Test check: valid probabilities
def test_check_valid_prob():
    df = pd.DataFrame({
        "feature": ["a", "b"],
        "class_0": [0.4, 0.2],
        "class_1": [0.6, 0.8],
    })
    seg = CategoricalSegmentationStats(df, label_col=["class_0", "class_1"], pred_col=None, feature="feature")
    seg.check()


def test_prepare_segments_qcut():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    seg = ContinuousSegmentationStats(df, label_col="label", pred_col="pred", feature="feature")
    df_out, segments = seg._prepare_segments(bins=3)
    assert len(segments) == 3


@pytest.mark.parametrize(
    "labels,expected_task",
    [
        ([0.1, 0.2, 0.3], "reg"),
        ([0, 1, 0], "binary"),
        ([0, 1, 2], "class"),
    ],
)
def test_get_metric_tasks(labels, expected_task):
    df = pd.DataFrame({"feature": [1, 2, 3], "label": labels, "pred": labels})
    seg = ContinuousSegmentationStats(df, label_col="label", pred_col="pred", feature="feature")
    metric, task = seg._get_metric(df["label"])
    assert task is not None


@pytest.mark.parametrize(
    "labels,preds,tasktype",
    [
        ([0, 1, 0, 1], [0, 1, 0, 1], "binary"),
        ([0, 1, 2, 1], [0, 1, 2, 1], "class"),
        ([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], "reg"),
    ],
)
def test_compute_segment_scores_all_tasks(labels, preds, tasktype):
    df = pd.DataFrame({"feature": [1, 1, 2, 2], "label": labels, "pred": preds})
    seg = ContinuousSegmentationStats(df, label_col="label", pred_col="pred", feature="feature")
    metric, task = seg._get_metric(df["label"])
    df_out, segments = seg._prepare_segments(bins=2)
    result = seg._compute_segment_scores(df_out, segments, metric, task)
    assert len(result) == 2


def test_check_label_col_nan():
    df = pd.DataFrame({"feature": [1, 2], "label": [1, None], "pred": [1, 0]})
    seg = ContinuousSegmentationStats(df, label_col="label", pred_col="pred", feature="feature")
    with pytest.raises(ValueError, match="NaN"):
        seg.check()


def test_run_regression():
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "label": [0.1, 0.2, 0.3, 0.4], "pred": [0.1, 0.2, 0.3, 0.4]})
    seg = ContinuousSegmentationStats(df, label_col="label", pred_col="pred", feature="feature")
    result = seg.run(bins=2)
    assert "segment" in result and "score" in result
