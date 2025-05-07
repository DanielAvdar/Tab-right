from typing import Any  # Add import for Any

import pandas as pd
import pytest

from tab_right.segmentations.calc_seg import SegmentationCalc

from .calc_check import CheckBaseSegmentationCalc


def make_example(
    data: dict,
    label_col: str = "label",
    prediction_col: str | list[str] = "prediction",
    segment_col: str = "segment_id",
    segment_names: dict[int, Any] | None = None,  # Add optional segment_names
) -> SegmentationCalc:
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame(data)
    gdf = df.groupby(segment_col)
    # Create segment_names map if not provided
    if segment_names is None:
        # Ensure keys are the actual group names (segment IDs) and cast to int for mypy
        segment_names = {int(name): name for name in gdf.groups.keys()}  # type: ignore[call-overload]
    # Pass the segment_names map to the constructor
    return SegmentationCalc(gdf, label_col, prediction_col, segment_names)


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

    def test_call_with_empty_group(self) -> None:
        """Test the __call__ method when one segment group is empty."""
        # Data where segment 3 exists in segment_names but has no rows
        data = {
            "segment_id": [1, 1, 2, 2],
            "label": [0, 1, 0, 1],
            "prediction": [0.1, 0.9, 0.2, 0.8],
        }
        # Explicitly define segment names including the empty segment 3
        segment_names = {1: "Group 1", 2: "Group 2", 3: "Group 3"}

        instance = make_example(data=data, segment_names=segment_names)
        metric = self.get_metric()
        result = instance(metric)

        # Check that all defined segments are present in the result
        assert len(result) == len(segment_names)
        assert set(result["segment_id"]) == set(segment_names.keys())

        # Check that the score for the empty segment (ID 3) is NaN
        empty_segment_score = result.loc[result["segment_id"] == 3, "score"].iloc[0]
        assert pd.isna(empty_segment_score)

        # Check that other segments have valid scores
        assert not result.loc[result["segment_id"] != 3, "score"].isnull().any()
