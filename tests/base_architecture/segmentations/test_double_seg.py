import pandas as pd
import pytest

from tab_right.segmentations.double_seg import DoubleSegmentationImp

from .double_check import CheckDoubleSegmentation


def make_example(data: dict, label_col: str = "label", prediction_col: str = "prediction") -> DoubleSegmentationImp:
    """Create a sample DataFrame for testing."""
    # Create DataFrame from input data
    df = pd.DataFrame(data)
    # Create and return a DoubleSegmentationImp instance
    return DoubleSegmentationImp(df=df, label_col=label_col, prediction_col=prediction_col)


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
            # Added test case with categorical features
            make_example(
                data={
                    "feature1": ["A", "B", "A", "C", "B", "C"],
                    "feature2": ["X", "Y", "Y", "X", "X", "Y"],
                    "label": [0, 1, 0, 1, 0, 1],
                    "prediction": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
                }
            ),
            # Added test case with one continuous and one categorical feature
            make_example(
                data={
                    "feature1": [1.1, 2.2, 1.5, 3.8, 2.9, 4.5],  # Continuous
                    "feature2": ["P", "Q", "P", "R", "Q", "R"],  # Categorical
                    "label": [0, 1, 0, 1, 0, 1],
                    "prediction": [0.15, 0.85, 0.25, 0.75, 0.35, 0.65],
                }
            ),
        ]
    )
    def instance_to_check(self, request: pytest.FixtureRequest) -> DoubleSegmentationImp:
        """Fixture to create parameterized instances of the class."""
        return request.param
