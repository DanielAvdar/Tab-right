import pandas as pd
import pytest

from tab_right.segmentations.calc_seg import SegmentationCalc

from ..seg_protocols_check import CheckBaseSegmentationCalc


def make_example(
    data: dict, label_col: str = "label", prediction_col: str | list[str] = "prediction"
) -> SegmentationCalc:
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
                    "segment_id": [1, 1, 2, 2, 3, 3],
                    "label": [0, 1, 0, 1, 0, 1],
                    "prob1": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
                    "prob2": [0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
                },
                prediction_col=["prob1", "prob2"],
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
    def instance_to_check(self, request: pytest.FixtureRequest) -> SegmentationCalc:
        """Fixture to create parameterized instances of the class."""
        return request.param
