import pandas as pd

from tab_right.base_architecture.seg_plotting_protocols import DoubleSegmPlottingP
from tests.base_architecture.seg_protocols_check import CheckProtocols


class CheckDoubleSegmPlotting(CheckProtocols):
    """Class for checking compliance of `DoubleSegmPlotting` protocol."""

    # Use the protocol type directly
    class_to_check = DoubleSegmPlottingP

    def test_attributes(self, instance_to_check: DoubleSegmPlottingP) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "df")
        assert hasattr(instance_to_check, "metric_name")
        assert isinstance(instance_to_check.df, pd.DataFrame)

    def test_get_heatmap_df(self, instance_to_check: DoubleSegmPlottingP) -> None:
        """Test the `get_heatmap_df` method of the instance."""
        result = instance_to_check.get_heatmap_df()
        assert isinstance(result, pd.DataFrame)
        # Verify the result is a pivot table format with feature_1 as columns and feature_2 as index
        assert len(result.columns) > 0
        assert len(result.index) > 0
        assert len(result.columns) == len(instance_to_check.df["feature_1"].unique())
        assert len(result.index) == len(instance_to_check.df["feature_2"].unique())
        assert result.isnull().sum().sum() < len(result) * len(result.columns)

    def test_plot_heatmap(self, instance_to_check: DoubleSegmPlottingP) -> None:
        """Test the `plot_heatmap` method of the instance."""
        from plotly.graph_objects import Figure, Heatmap

        result = instance_to_check.plot_heatmap()
        assert isinstance(result, Figure)

        # Verify the figure contains a heatmap trace
        assert len(result.data) > 0
        assert any(isinstance(trace, Heatmap) for trace in result.data)

        # Check that the layout includes appropriate axis titles
        assert result.layout.xaxis.title is not None
        assert result.layout.yaxis.title is not None
