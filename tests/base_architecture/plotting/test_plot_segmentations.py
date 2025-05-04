"""Tests for the plot_segmentations module."""

import pandas as pd
import pytest

from tab_right.plotting.plot_segmentations import (
    DoubleSegmPlotting,
)
from tests.base_architecture.seg_protocols_check import CheckDoubleSegmPlotting


class TestDoubleSegmPlotting(CheckDoubleSegmPlotting):
    """Test class for double segmentation plotting."""

    @pytest.fixture
    def double_segmentation_df(self):
        """Create a sample DataFrame for testing double segmentation plotting."""
        return pd.DataFrame({
            "segment_id": [1, 2, 3, 4, 5, 6],
            "feature_1": ["A <= 10", "A <= 10", "10 < A <= 20", "10 < A <= 20", "A > 20", "A > 20"],
            "feature_2": ["B <= 5", "B > 5", "B <= 5", "B > 5", "B <= 5", "B > 5"],
            "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })

    @pytest.fixture
    def instance_to_check(self, double_segmentation_df):
        """Create an instance of DoubleSegmPlotting for testing."""
        return DoubleSegmPlotting(df=double_segmentation_df)
