import pandas as pd
import pytest

from tab_right.seg import SegmentationStats


@pytest.mark.parametrize(
    "df,label_col,pred_col,feature",
    [
        (
            pd.DataFrame({"feature1": [1, 2, 1, 2], "label": [0, 1, 1, 0], "prediction": [0, 1, 1, 0]}),
            "label",
            "prediction",
            "feature1",
        ),
        (
            pd.DataFrame({"cat": ["a", "b", "a", "b"], "label": [1, 0, 1, 0], "prediction": [1, 0, 1, 0]}),
            "label",
            "prediction",
            "cat",
        ),
        (
            pd.DataFrame({"cont": [0.1, 0.2, 0.3, 0.4], "label": [1, 0, 1, 0], "prediction": [1, 0, 1, 0]}),
            "label",
            "prediction",
            "cont",
        ),
    ],
)
@pytest.mark.parametrize(
    "arrow",
    [
        True,
        False,

        ]
)
def test_segmentation_stats_run(df, label_col, pred_col, feature, arrow):
    if arrow:
        from pandas_pyarrow import convert_to_pyarrow
        df = convert_to_pyarrow(df)
    seg = SegmentationStats(df, label_col=label_col, pred_col=pred_col, feature=feature)
    result = seg.run()
    assert list(result.columns) == ["segment", "score"]
    assert result.shape[0] > 0
