import pandas as pd
import pytest

from tab_right.segmentations.double_seg import DoubleSegmentationImp
from tab_right.segmentations.find_seg import FindSegmentationImp

from ..seg_protocols_check import CheckDoubleSegmentation


def make_example(data: dict, label_col: str = "label", prediction_col: str = "prediction") -> DoubleSegmentationImp:
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame(data)
    segmentation_finder = FindSegmentationImp(df, label_col, prediction_col)
    return DoubleSegmentationImp(segmentation_finder)


class TestDoubleSegmentationImp(CheckDoubleSegmentation):
    """Test class for double segmentation."""

    @pytest.fixture(
        params=[
            make_example(
                data={
                    "feature1": [1, 2, 3, 4],
                    "feature2": [5, 6, 7, 8],
                    "label": [0, 1, 0, 1],
                    "prediction": [0.1, 0.9, 0.2, 0.8],
                }
            ),
            make_example(
                data={
                    "feature1": [1, 2, 3, 4, 5, 6],
                    "feature2": [5, 6, 7, 8, 9, 10],
                    "label": [0, 1, 0, 1, 0, 1],
                    "prediction": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
                }
            ),
            make_example(
                data={
                    "feature1": [1, 2, 3, 4],
                    "feature2": [5, 6, 7, 8],
                    "label": [0, 1, 0, 1],
                    "prediction": [0.1, 0.9, 0.2, 0.8],
                    "score": [0.2, 0.8, 0.3, 0.7],
                },
                prediction_col="score",
            ),
        ]
    )
    def instance_to_check(self, request: pytest.FixtureRequest) -> DoubleSegmentationImp:
        """Fixture to create parameterized instances of the class."""
        return request.param
