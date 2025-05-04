"""Module for checking segmentation protocols."""

from typing import Any, Callable

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from sklearn.metrics import log_loss

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
        assert "name" in result.columns  # Added assertion for 'name' column
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

    def test_group_2_features(self, instance_to_check: Any) -> None:
        """Test the _group_2_features method functionality."""
        # Define dummy feature names and bin counts based on typical test data
        feature1_col = "feature1"  # Assumes 'feature1' exists in the test instance's df
        feature2_col = "feature2"  # Assumes 'feature2' exists in the test instance's df
        bins_1 = 4
        bins_2 = 4

        # Call the internal grouping method
        calc_instance = instance_to_check._group_2_features(feature1_col, feature2_col, bins_1, bins_2)

        # Check if the returned object conforms to the BaseSegmentationCalc protocol
        assert isinstance(calc_instance, BaseSegmentationCalc)
        assert hasattr(calc_instance, "gdf")
        assert hasattr(calc_instance, "label_col")
        assert hasattr(calc_instance, "prediction_col")
        assert isinstance(calc_instance.gdf, DataFrameGroupBy)

        # Check if the number of groups is reasonable (less than or equal to bins_1 * bins_2 or unique combos)
        unique_f1 = instance_to_check.df[feature1_col].nunique()
        unique_f2 = instance_to_check.df[feature2_col].nunique()
        max_expected_groups = min(bins_1, unique_f1) * min(bins_2, unique_f2)  # Approximation
        assert len(calc_instance.gdf.groups) <= max_expected_groups
        assert len(calc_instance.gdf.groups) > 0  # Should have at least one group

    def test_call(
        self,
        instance_to_check: Any,
    ) -> None:
        """Test the `__call__` method of the instance."""
        # Use the aggregated metric function as required by the protocol's __call__ signature
        score_metric = self.get_metric(agg=True)
        # Define dummy feature names and bin counts
        feature1_col = "feature1"  # Assuming 'feature1' exists in the test instance's df
        feature2_col = "feature2"  # Assuming 'feature2' exists in the test instance's df
        bins_1 = 4
        bins_2 = 4

        # Call the instance with the correct arguments: feature cols, metric, and bins
        result = instance_to_check(feature1_col, feature2_col, score_metric, bins_1, bins_2)  # Corrected arguments
        assert isinstance(result, pd.DataFrame)
        assert "segment_id" in result.columns
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "score" in result.columns
        # Ensure no NaN values in the final score output
        assert not result["score"].isnull().any()
        # Ensure features columns correctly reflect the grouping
        unique_f1 = instance_to_check.df[feature1_col].nunique()
        unique_f2 = instance_to_check.df[feature2_col].nunique()
        # Calculate max expected groups more accurately based on binning logic
        max_expected_groups_f1 = (
            min(bins_1, unique_f1) if pd.api.types.is_numeric_dtype(instance_to_check.df[feature1_col]) else unique_f1
        )
        max_expected_groups_f2 = (
            min(bins_2, unique_f2) if pd.api.types.is_numeric_dtype(instance_to_check.df[feature2_col]) else unique_f2
        )
        max_expected_groups = max_expected_groups_f1 * max_expected_groups_f2

        assert result["segment_id"].nunique() <= max_expected_groups
        assert result["segment_id"].nunique() > 0


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
