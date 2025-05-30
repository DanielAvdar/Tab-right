import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from tab_right.base_architecture.seg_protocols import BaseSegmentationCalc, DoubleSegmentation
from tests.base_architecture.base_protocols_check import CheckProtocols


class CheckDoubleSegmentation(CheckProtocols):
    """Class for checking compliance of `DoubleSegmentation` protocol."""

    # Use the protocol type directly
    class_to_check = DoubleSegmentation

    def test_attributes(self, instance_to_check: DoubleSegmentation) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "df")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_group_2_features(self, instance_to_check: DoubleSegmentation) -> None:
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
        instance_to_check: DoubleSegmentation,
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
