"""Tests for the plot_segmentations module."""

import pandas as pd
import pytest
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objects import Figure as PlotlyFigure

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
            {
                "df": pd.DataFrame({
                    "segment_id": [1, 2, 3, 4, 5, 6],
                    "feature_1": ["[0,1]", "[1,2]", "[2,3]", "[3,4]", "[4,5]", "[5,6]"],
                    "feature_2": ["cat_A", "cat_B", "cat_A", "cat_B", "cat_A", "cat_B"],
                    "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                }),
                "backend": "plotly",
            },
            {
                "df": pd.DataFrame({
                    "segment_id": [1, 2, 3, 4, 5, 6],
                    "feature_1": ["[0,1]", "[1,2]", "[2,3]", "[3,4]", "[4,5]", "[5,6]"],
                    "feature_2": ["cat_A", "cat_B", "cat_A", "cat_B", "cat_A", "cat_B"],
                    "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                }),
                "backend": "matplotlib",
            },
            {
                "df": pd.DataFrame({
                    "segment_id": [1, 2, 3, 4, 5, 6],
                    "feature_1": ["[0.3,0.4]", "[0.4,0.5]", "[0.5,0.6]", "[0.6,0.7]", "[0.7,0.8]", "[0.8,0.9]"],
                    "feature_2": ["A", "B", "A", "B", "A", "B"],
                    "score": [0.0, 0.5, 0.6, 0.7, 0.1, 0.2],
                }),
                "backend": "plotly",
            },
            {
                "df": pd.DataFrame({
                    "segment_id": [1, 2, 3, 4, 5, 6],
                    "feature_1": ["[0.3,0.4]", "[0.4,0.5]", "[0.5,0.6]", "[0.6,0.7]", "[0.7,0.8]", "[0.8,0.9]"],
                    "feature_2": ["A", "B", "A", "B", "A", "B"],
                    "score": [0.0, 0.5, 0.6, 0.7, 0.1, 0.2],
                }),
                "backend": "matplotlib",
            },
        ]
    )
    def instance_to_check(
        self,
        request: pytest.FixtureRequest,
    ) -> DoubleSegmPlotting:
        """Create an instance of DoubleSegmPlotting for testing."""
        params = request.param
        return DoubleSegmPlotting(df=params["df"], backend=params["backend"])

    def test_backend_selection(self, double_segmentation_df: pd.DataFrame) -> None:
        """Test that the backend selection works correctly."""
        # Test Plotly backend
        plotly_instance = DoubleSegmPlotting(df=double_segmentation_df, backend="plotly")
        plotly_result = plotly_instance.plot_heatmap()
        assert isinstance(plotly_result, PlotlyFigure)

        # Test Matplotlib backend
        matplotlib_instance = DoubleSegmPlotting(df=double_segmentation_df, backend="matplotlib")
        matplotlib_result = matplotlib_instance.plot_heatmap()
        assert isinstance(matplotlib_result, MatplotlibFigure)
