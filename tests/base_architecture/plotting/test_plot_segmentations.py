"""Tests for the plot_segmentations module."""

import pandas as pd
import pytest

from tab_right.plotting.plot_segmentations import (
    DoubleSegmPlotting,
)
from tests.base_architecture.plotting.seg_plot_check import CheckDoubleSegmPlotting


class TestDoubleSegmPlotting(CheckDoubleSegmPlotting):
    """Test class for double segmentation plotting."""

    @pytest.fixture
    def double_segmentation_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing double segmentation plotting."""
        return pd.DataFrame({
            "segment_id": [1, 2, 3, 4, 5, 6],
            "feature_1": ["[0,1]", "[1,2]", "[2,3]", "[3,4]", "[4,5]", "[5,6]"],
            "feature_2": ["cat_A", "cat_B", "cat_A", "cat_B", "cat_A", "cat_B"],
            "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })

    @pytest.fixture(
        params=[
            pd.DataFrame({
                "segment_id": [1, 2, 3, 4, 5, 6],
                "feature_1": ["[0,1]", "[1,2]", "[2,3]", "[3,4]", "[4,5]", "[5,6]"],
                "feature_2": ["cat_A", "cat_B", "cat_A", "cat_B", "cat_A", "cat_B"],
                "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }),
            pd.DataFrame({
                "segment_id": [1, 2, 3, 4, 5, 6],
                "feature_1": ["[0.3,0.4]", "[0.4,0.5]", "[0.5,0.6]", "[0.6,0.7]", "[0.7,0.8]", "[0.8,0.9]"],
                "feature_2": ["A", "B", "A", "B", "A", "B"],
                "score": [0.0, 0.5, 0.6, 0.7, 0.1, 0.2],
            }),
        ]
    )
    def instance_to_check(
        self,
        request: pytest.FixtureRequest,
    ) -> DoubleSegmPlotting:
        """Create an instance of DoubleSegmPlotting for testing."""
        double_segmentation_df = request.param
        return DoubleSegmPlotting(df=double_segmentation_df)
