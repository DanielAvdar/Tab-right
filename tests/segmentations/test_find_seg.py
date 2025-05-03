import pandas as pd
import pytest

from tab_right.base_architecture.seg_protocols_check import CheckFindSegmentation
from tab_right.segmentations.find_seg import FindSegmentationImp


def make_example(data, prediction_col):
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame(data)
    return FindSegmentationImp(df, "label", prediction_col)


class TestFindSegmentationImp(CheckFindSegmentation):
    """Test class for double segmentation."""

    @pytest.fixture(
        params=[
            make_example(
                data={"feature": [1, 2, 3, 4], "label": [0, 1, 0, 1], "prediction": [0.1, 0.9, 0.2, 0.8]},
                prediction_col="prediction",
            ),
            make_example(
                data={
                    "feature": [1, 2, 3, 4],
                    "label": [0, 1, 0, 1],
                    "prediction1": [0.1, 0.9, 0.2, 0.8],
                    "prediction2": [0.2, 0.8, 0.3, 0.7],
                },
                prediction_col=["prediction1", "prediction2"],
            ),
            make_example(
                data={
                    "feature": [1, 2, 3, 4],
                    "label": [0, 1, 0, 1],
                    "prediction1": [0.1, 0.9, 0.2, 0.8],
                    "prediction2": [0.2, 0.8, 0.3, 0.7],
                },
                prediction_col="prediction1",
            ),
        ]
    )
    def instance_to_check(self, request):
        return request.param
