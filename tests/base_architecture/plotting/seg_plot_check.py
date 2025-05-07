import pandas as pd

from tab_right.base_architecture.seg_plotting_protocols import DoubleSegmPlottingP

from ..base_protocols_check import CheckProtocols


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
        from matplotlib.figure import Figure as MatplotlibFigure
        from plotly.graph_objects import Figure as PlotlyFigure, Heatmap

        result = instance_to_check.plot_heatmap()

        # Check that the result is either a Plotly Figure or a Matplotlib Figure
        assert isinstance(result, (PlotlyFigure, MatplotlibFigure)), (
            "Result must be either a Plotly or Matplotlib figure"
        )

        if isinstance(result, PlotlyFigure):
            # Verify the plotly figure contains a heatmap trace
            assert len(result.data) > 0
            assert any(isinstance(trace, Heatmap) for trace in result.data)

            # Check that the layout includes appropriate axis titles
            assert result.layout.xaxis.title is not None
            assert result.layout.yaxis.title is not None

        elif isinstance(result, MatplotlibFigure):
            # For matplotlib, verify that the figure contains at least one axes
            assert len(result.axes) > 0

            # Verify that the axes has a title and axis labels
            ax = result.axes[0]
            assert ax.get_xlabel() != ""
            assert ax.get_ylabel() != ""

            # Verify there's a colorbar or a collection (like a heatmap)
            assert any(collection.__class__.__name__ == "QuadMesh" for collection in ax.collections), (
                "No heatmap found in the matplotlib figure"
            )
