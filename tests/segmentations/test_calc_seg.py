import pandas as pd
import pytest

from tab_right.base_architecture.seg_protocols_check import CheckBaseSegmentationCalc
from tab_right.segmentations.calc_seg import SegmentationCalc


def make_example(data, label_col="label", prediction_col="prediction"):
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame(data)
    gdf = df.groupby("segment_id")
    return SegmentationCalc(gdf, label_col, prediction_col)


class TestBaseSegmentationCalcImp(CheckBaseSegmentationCalc):
    """Test class for double segmentation."""

    @pytest.fixture(
        params=[
            make_example(
                data={
                    "segment_id": [1, 1, 2, 2],
                    "label": [0, 1, 0, 1],
                    "prediction": [0.1, 0.9, 0.2, 0.8],
                }
            ),
            make_example(
                data={
                    "segment_id": [1, 1, 2, 2, 3, 3],
                    "label": [0, 1, 0, 1, 0, 1],
                    "prediction": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
                }
            ),
            make_example(
                data={
                    "segment_id": [1, 1, 2, 2],
                    "label": [0, 1, 0, 1],
                    "prediction": [0.1, 0.9, 0.2, 0.8],
                    "score": [0.2, 0.8, 0.3, 0.7],
                },
                prediction_col="score",
            ),
        ]
    )
    def instance_to_check(self, request):
        """Fixture to create parameterized instances of the class."""
        return request.param
