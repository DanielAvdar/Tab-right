import pandas as pd
from sklearn.metrics import log_loss

from tab_right.base_architecture.seg_protocols import BaseSegmentationCalc
from tests.base_architecture.base_protocols_check import CheckProtocols


class CheckBaseSegmentationCalc(CheckProtocols):
    """Class for checking compliance of `BaseSegmentationCalc` protocol."""

    # Use the protocol type directly
    class_to_check = BaseSegmentationCalc

    def test_attributes(self, instance_to_check: BaseSegmentationCalc) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "gdf")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")
        assert hasattr(instance_to_check, "segment_names")  # Check optional attribute exists

    def test_reduce_metric_results(self, instance_to_check: BaseSegmentationCalc) -> None:
        """Test the metric reduction method of the instance."""
        series_results = pd.Series([0.1, 0.2, 0.3, 0.4])
        reduced_result = instance_to_check._reduce_metric_results(series_results)
        assert isinstance(reduced_result, float)
        assert reduced_result == series_results.mean()
        float_result: float = 0.5
        reduced_result = instance_to_check._reduce_metric_results(float_result)
        assert isinstance(reduced_result, float)
        assert reduced_result == float_result

    def test_call(self, instance_to_check: BaseSegmentationCalc) -> None:
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
