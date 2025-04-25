import pandas as pd

from tab_right.seg import SegmentationStats


def test_segmentation_stats_basic():
    df = pd.DataFrame({"feature1": [1, 2, 1, 2], "label": [0, 1, 1, 0], "prediction": [0, 1, 1, 0]})
    seg = SegmentationStats(df, label_col="label", pred_col="prediction")
    result = seg.run()
    assert "count" in result.columns
    assert "mean_label" in result.columns
    assert "mean_pred" in result.columns
    assert "std_pred" in result.columns
    assert "accuracy" in result.columns
    assert result.shape[0] == df["prediction"].nunique()
