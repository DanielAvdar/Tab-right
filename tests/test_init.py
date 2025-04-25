import pandas as pd
import pytest
from tab_right.seg import SegmentationStats

@pytest.mark.parametrize(
    "df,label_col,pred_col",
    [
        (pd.DataFrame({"feature1": [1, 2, 1, 2], "label": [0, 1, 1, 0], "prediction": [0, 1, 1, 0]}), "label", "prediction"),
        (pd.DataFrame({"feature1": [1, 2, 3], "label": [1, 0, 1], "prediction": [1, 0, 1]}), "label", "prediction"),
        (pd.DataFrame({"feature1": [1, 2, 3], "target": [1, 2, 3], "pred": [1, 2, 3]}), "target", "pred"),
    ]
)
def test_segmentation_stats_param(df, label_col, pred_col):
    seg = SegmentationStats(df, label_col=label_col, pred_col=pred_col)
    result = seg.run()
    assert "count" in result.columns
    assert "mean_label" in result.columns
    assert "mean_pred" in result.columns
    assert "std_pred" in result.columns
    # Only check accuracy if binary
    if set(df[label_col].dropna().unique()).issubset({0, 1}):
        assert "accuracy" in result.columns
    assert result.shape[0] == df[pred_col].nunique()
