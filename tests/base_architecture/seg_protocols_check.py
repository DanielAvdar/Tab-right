"""Module for checking segmentation protocols."""

from typing import Any, Callable

import pandas as pd
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeRegressor

from tab_right.base_architecture.seg_plotting_protocols import DoubleSegmPlottingP
from tab_right.base_architecture.seg_protocols import BaseSegmentationCalc, DoubleSegmentation


class CheckProtocols:
    """Base class for checking protocol compliance."""

    class_to_check: Any = None

    def get_metric(self, agg: bool = False) -> Callable:
        def metric_single(y: pd.Series, p: pd.Series) -> pd.Series:
            return abs(y - p)

        def agg_metric_single(y: pd.Series, p: pd.Series) -> float:
            return float(abs(y - p).mean())

        if agg:
            return agg_metric_single
        return metric_single

    def test_protocol_followed(self, instance_to_check: Any) -> None:
        """Test if the protocol is followed correctly."""
        # check if it is dataclass
        assert hasattr(instance_to_check, "__dataclass_fields__")
        assert isinstance(instance_to_check, self.class_to_check)


class CheckBaseSegmentationCalc(CheckProtocols):
    """Class for checking compliance of `BaseSegmentationCalc` protocol."""

    # Use the protocol type directly
    class_to_check = BaseSegmentationCalc

    def test_attributes(self, instance_to_check: Any) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "gdf")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_reduce_metric_results(self, instance_to_check: Any) -> None:
        """Test the metric reduction method of the instance."""
        series_results = pd.Series([0.1, 0.2, 0.3, 0.4])
        reduced_result = instance_to_check._reduce_metric_results(series_results)
        assert isinstance(reduced_result, float)
        assert reduced_result == series_results.mean()
        float_result: float = 0.5
        reduced_result = instance_to_check._reduce_metric_results(float_result)
        assert isinstance(reduced_result, float)
        assert reduced_result == float_result

    def test_call(self, instance_to_check: Any) -> None:
        """Test the `__call__` method of the instance."""
        metric = self.get_metric() if isinstance(instance_to_check.prediction_col, str) else log_loss
        result = instance_to_check(metric)
        assert "segment_id" in result.columns
        assert "score" in result.columns
        number_of_groups = len(instance_to_check.gdf.groups)
        number_of_segments = len(result)
        assert number_of_groups == number_of_segments
        assert not result.isnull().values.any()


class CheckDoubleSegmentation(CheckProtocols):
    """Class for checking compliance of `DoubleSegmentation` protocol."""

    # Use the protocol type directly
    class_to_check = DoubleSegmentation

    def test_attributes(self, instance_to_check: Any) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "df")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_call(
        self,
        instance_to_check: Any,
    ) -> None:
        """Test the `__call__` method of the instance."""
        model = DecisionTreeRegressor()
        error_func = self.get_metric()
        metric = self.get_metric(agg=True)

        result = instance_to_check("feature1", "feature2", error_func, model, metric)
        assert "segment_id" in result.columns
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "score" in result.columns

    def test_combine_2_features(self, instance_to_check: Any) -> None:
        """Test the method that combines features from two dataframes.

        Verifies that feature columns are properly combined and segment IDs are preserved.
        """
        segment_id1 = list(range(6))
        segment_name1 = ["A", "B", "C", "D", "E", "F"]
        score1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        segment_id2 = list(range(6, 12))
        segment_name2 = ["G", "H", "I", "J", "K", "L"]
        score2 = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        df1 = pd.DataFrame({"segment_id": segment_id1, "feature_1": segment_name1, "score_1": score1})
        df2 = pd.DataFrame({"segment_id": segment_id2, "feature_2": segment_name2, "score_2": score2})
        combined = instance_to_check._combine_2_features(df1, df2)
        assert "feature_1" in combined.columns
        assert "feature_2" in combined.columns
        assert len(combined) == len(df1)
        assert len(combined) == len(df2)
        assert "score" in combined.columns
        assert combined["segment_id"].equals(pd.Series(range(6)))
        assert combined.isnull().sum().sum() == 0

    def test_group_by_segment(self, instance_to_check: Any) -> None:
        """Test the group_by_segment method functionality.

        Ensures the method correctly groups a DataFrame by segment ID and returns a DataFrameGroupBy object.
        """
        df = pd.DataFrame({"segment_id": [1, 1, 2, 2], "score": [0.1, 0.2, 0.3, 0.4]})
        seg = df["segment_id"]
        bsc = instance_to_check._group_by_segment(df, seg)
        assert isinstance(bsc, BaseSegmentationCalc)
        assert hasattr(bsc, "gdf")
        total_len = len(bsc.gdf.groups)
        assert total_len == 2


class CheckDoubleSegmPlotting(CheckProtocols):
    """Class for checking compliance of `DoubleSegmPlotting` protocol."""

    # Use the protocol type directly
    class_to_check = DoubleSegmPlottingP

    def test_attributes(self, instance_to_check: Any) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "df")
        assert hasattr(instance_to_check, "metric_name")
        assert isinstance(instance_to_check.df, pd.DataFrame)

    def test_get_heatmap_df(self, instance_to_check: Any) -> None:
        """Test the `get_heatmap_df` method of the instance."""
        result = instance_to_check.get_heatmap_df()
        assert isinstance(result, pd.DataFrame)
        # Verify the result is a pivot table format with feature_1 as columns and feature_2 as index
        assert len(result.columns) > 0
        assert len(result.index) > 0
        assert len(result.columns) == len(instance_to_check.df["feature_1"].unique())
        assert len(result.index) == len(instance_to_check.df["feature_2"].unique())
        assert result.isnull().sum().sum() < len(result) * len(result.columns)

    def test_plotly_heatmap(self, instance_to_check: Any) -> None:
        """Test the `plotly_heatmap` method of the instance."""
        from plotly.graph_objects import Figure, Heatmap

        result = instance_to_check.plotly_heatmap()
        assert isinstance(result, Figure)

        # Verify the figure contains a heatmap trace
        assert len(result.data) > 0
        assert any(isinstance(trace, Heatmap) for trace in result.data)

        # Check that the layout includes appropriate axis titles
        assert result.layout.xaxis.title is not None
        assert result.layout.yaxis.title is not None
