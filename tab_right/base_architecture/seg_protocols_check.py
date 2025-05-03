"""Module for checking segmentation protocols."""

import abc
from typing import Any, Callable, Union

import pandas as pd
import pytest
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from .seg_plotting_protocols import DoubleSegmPlotting
from .seg_protocols import BaseSegmentationCalc, DoubleSegmentation, FindSegmentation


class CheckProtocols:
    """Base class for checking protocol compliance."""

    class_to_check = None  # type: ignore[assignment]

    @abc.abstractmethod
    @pytest.fixture
    def instance_to_check(self) -> Any:
        """Fixture to create an instance of the class."""

    def test_protocol_followed(self, instance_to_check: Any) -> None:
        """Test if the protocol is followed correctly."""
        # check if it is dataclass
        assert hasattr(instance_to_check, "__dataclass_fields__")
        assert isinstance(instance_to_check, self.class_to_check)

    def get_metric(self, prediction_col: Union[str, list[str]]) -> Callable:
        """Get scikit-learn metric function based on the prediction column.

        Args:
            prediction_col: Column name(s) for predictions
            agg: Whether to return an aggregated metric or per-sample errors

        Returns:
            Callable metric function that handles both single and multiple predictions

        """

        # For non-aggregated metrics (return error per sample)
        def metric_single(y, p):
            return abs(y - p)

        return metric_single


class CheckFindSegmentation(CheckProtocols):
    """Class for checking compliance of `FindSegmentation` protocol."""

    class_to_check = FindSegmentation

    def test_attributes(self, instance_to_check: Any) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "df")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_call(self, instance_to_check: Any) -> None:
        """Test the `__call__` method of the instance."""
        model = DecisionTreeRegressor()
        metric = self.get_metric(instance_to_check.prediction_col)
        result = instance_to_check("feature", metric, model)
        assert "segment_id" in result.columns
        assert "segment_name" in result.columns
        assert "score" in result.columns

    def test_calc_error(self, instance_to_check: Any) -> None:
        """Test the error calculation method of the instance."""
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.Series([0.1, 0.9, 0.2, 0.8])

        metric = self.get_metric(
            "prediction",
        )

        result = instance_to_check._calc_error(metric, y_true, y_pred)
        assert len(result) == len(y_true)
        assert isinstance(result, pd.Series)
        assert metric(y_true, y_pred).equals(result)

    def test_fit_model(self, instance_to_check):
        """Test the model fitting method of the instance."""
        feature = pd.Series([1, 2, 3, 4])
        error = pd.Series([0.1, 0.2, 0.3, 0.4])
        model = DecisionTreeRegressor()
        fitted_model = instance_to_check._fit_model(model, feature, error)
        assert hasattr(fitted_model, "tree_")

    def test_extract_leaves(self, instance_to_check):
        """Test the leaf extraction method of the instance."""
        model = DecisionTreeRegressor(max_depth=2)
        feature = pd.Series([1, 2, 3, 4])
        error = pd.Series([0.1, 0.2, 0.3, 0.4])
        model.fit(feature.values.reshape(-1, 1), error)
        leaves = instance_to_check._extract_leaves(model)
        assert "segment_id" in leaves.columns
        assert "segment_name" in leaves.columns


class CheckBaseSegmentationCalc(CheckProtocols):
    """Class for checking compliance of `BaseSegmentationCalc` protocol."""

    class_to_check = BaseSegmentationCalc

    def get_metric(self, prediction_col: Union[str, list[str]]) -> Callable:
        """Get scikit-learn metric function based on the prediction column.

        Args:
            prediction_col: Column name(s) for predictions
            agg: Whether to return an aggregated metric or per-sample errors

        Returns:
            Callable metric function that handles both single and multiple predictions

        """
        # For non-aggregated metrics (return error per sample)
        if isinstance(prediction_col, list):
            return log_loss

        return super().get_metric(prediction_col)

    def test_attributes(self, instance_to_check: Any) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "gdf")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_reduce_metric_results(self, instance_to_check: Any) -> None:
        """Test the metric reduction method of the instance."""
        results = pd.Series([0.1, 0.2, 0.3, 0.4])
        reduced_result = instance_to_check._reduce_metric_results(results)
        assert isinstance(reduced_result, float)
        assert reduced_result == results.mean()
        results = 0.5
        reduced_result = instance_to_check._reduce_metric_results(results)
        assert isinstance(reduced_result, float)
        assert reduced_result == results

    def test_call(self, instance_to_check: Any) -> None:
        """Test the `__call__` method of the instance."""
        metric = self.get_metric(instance_to_check.prediction_col)
        result = instance_to_check(metric)
        assert "segment_id" in result.columns
        assert "score" in result.columns
        number_of_groups = len(instance_to_check.gdf.groups)
        number_of_segments = len(result)
        assert number_of_groups == number_of_segments


class CheckDoubleSegmentation(CheckProtocols):
    """Class for checking compliance of `DoubleSegmentation` protocol."""

    class_to_check = DoubleSegmentation

    @pytest.fixture(
        params=[
            log_loss,
            mean_absolute_error,
            mean_squared_error,
        ]
    )
    def skl_metric(self, request) -> Callable:
        """Fixture to create parameterized instances of the class."""
        return request.param

    def test_attributes(self, instance_to_check: Any) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "segmentation_finder")
        assert isinstance(instance_to_check.segmentation_finder, FindSegmentation)

    def test_call(self, instance_to_check: Any) -> None:
        """Test the `__call__` method of the instance."""
        model = DecisionTreeRegressor()
        metric = self.get_metric(instance_to_check.segmentation_finder.prediction_col)
        score_metric = lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
        result = instance_to_check("feature1", "feature2", metric, model, score_metric)
        assert "segment_id" in result.columns
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "score" in result.columns

    def test_combine_2_features(self, instance_to_check):
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

    def test_group_by_segment(self, instance_to_check):
        """Test the group_by_segment method functionality.

        Ensures the method correctly groups a DataFrame by segment ID and returns a DataFrameGroupBy object.
        """
        df = pd.DataFrame({"segment_id": [1, 1, 2, 2], "score": [0.1, 0.2, 0.3, 0.4]})
        seg = df["segment_id"]
        bsc = instance_to_check._group_by_segment(df, seg)
        assert isinstance(bsc, BaseSegmentationCalc)


class CheckDoubleSegmPlotting(CheckProtocols):
    """Class for checking compliance of `DoubleSegmPlotting` protocol."""

    class_to_check = DoubleSegmPlotting

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
        # The values in the DataFrame should be the metric scores
        assert result.values.dtype.kind in "fiub"  # Numeric types (float, int, uint, bool)

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
