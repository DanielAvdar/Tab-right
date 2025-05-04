import pandas as pd
import pytest

from tab_right.segmentations.find_seg import FindSegmentationImp

from ..seg_protocols_check import CheckFindSegmentation


def make_example(data: dict, prediction_col: str) -> FindSegmentationImp:
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
                    "feature": [1, 2, 3, 4, 5, 6],
                    "label": [0, 1, 0, 1, 0, 1],
                    "prediction": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
                },
                prediction_col="prediction",
            ),
            make_example(
                data={"feature": [-1, -2, -3, -4], "label": [0, 1, 0, 1], "prediction": [0.1, 0.9, 0.2, 0.8]},
                prediction_col="prediction",
            ),
            make_example(
                data={"feature": [-1, -2, -3, -4], "label": [0, 1, 0, 1], "prediction": [-0.1, -0.9, -0.2, -0.8]},
                prediction_col="prediction",
            ),
        ]
    )
    def instance_to_check(self, request: pytest.FixtureRequest) -> FindSegmentationImp:
        """Fixture to create parameterized instances of the class."""
        return request.param
