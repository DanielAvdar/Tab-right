import numpy as np
import pandas as pd

from tab_right.base_architecture.drift_protocols import DriftCalcP

from ..base_protocols_check import CheckProtocols


class CheckDriftCalc(CheckProtocols):
    """Class for checking compliance of `DriftCalc` protocol."""

    # Use the protocol type directly
    class_to_check = DriftCalcP

    def test_attributes(self, instance_to_check: DriftCalcP) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "df1")
        assert hasattr(instance_to_check, "df2")
        assert hasattr(instance_to_check, "kind")
        assert isinstance(instance_to_check.df1, pd.DataFrame)
        assert isinstance(instance_to_check.df2, pd.DataFrame)

    def test_call_method(self, instance_to_check: DriftCalcP) -> None:
        """Test the __call__ method of the instance."""
        # Test with default parameters
        result = instance_to_check()
        assert isinstance(result, pd.DataFrame)
        # Check required columns
        assert "feature" in result.columns
        assert "type" in result.columns
        assert "score" in result.columns
        # Check score ranges
        assert (result["score"] >= 0).all() and (result["score"] <= 1).all()

        # Test with specific columns
        common_columns = set(instance_to_check.df1.columns) & set(instance_to_check.df2.columns)
        if len(common_columns) > 1:
            test_columns = list(common_columns)[:2]  # Just take 2 columns for testing
            result_subset = instance_to_check(columns=test_columns)
            assert isinstance(result_subset, pd.DataFrame)
            assert set(result_subset["feature"]) == set(test_columns)

        # Test with different bins
        result_bins = instance_to_check(bins=5)
        assert isinstance(result_bins, pd.DataFrame)
        assert "feature" in result_bins.columns

    def test_get_prob_density(self, instance_to_check: DriftCalcP) -> None:
        """Test the get_prob_density method of the instance."""
        result = instance_to_check.get_prob_density()
        assert isinstance(result, pd.DataFrame)
        # Check required columns
        assert "feature" in result.columns
        assert "bin" in result.columns
        assert "ref_density" in result.columns
        assert "cur_density" in result.columns
        # Check that densities sum to approximately 1 for each feature
        for feature in result["feature"].unique():
            feature_data = result[result["feature"] == feature]
            assert abs(feature_data["ref_density"].sum() - 1.0) < 0.01
            assert abs(feature_data["cur_density"].sum() - 1.0) < 0.01

    def test_categorical_drift_calc(self, instance_to_check: DriftCalcP) -> None:
        """Test the _categorical_drift_calc classmethod."""
        # Create test series
        s1 = pd.Series(["a", "b", "a", "c", "b", "a"])
        s2 = pd.Series(["a", "c", "c", "c", "b", "a"])

        # Calculate drift
        drift = instance_to_check.__class__._categorical_drift_calc(s1, s2)

        # Check output
        assert isinstance(drift, float)
        assert 0 <= drift <= 1, "Drift score should be between 0 and 1"

        # Check identical distributions
        s_identical = pd.Series(["a", "b", "c", "a", "b", "c"])
        drift_identical = instance_to_check.__class__._categorical_drift_calc(s_identical, s_identical)
        assert drift_identical == 0 or drift_identical < 0.01, "Identical distributions should have drift close to 0"

    def test_continuous_drift_calc(self, instance_to_check: DriftCalcP) -> None:
        """Test the _continuous_drift_calc classmethod."""
        # Create test series
        s1 = pd.Series(np.random.normal(0, 1, 100))
        s2 = pd.Series(np.random.normal(2, 1, 100))  # Different mean

        # Calculate drift
        drift = instance_to_check.__class__._continuous_drift_calc(s1, s2, bins=10)

        # Check output
        assert isinstance(drift, float)
        assert 0 <= drift <= 1, "Drift score should be between 0 and 1"

        # Check identical distributions
        drift_identical = instance_to_check.__class__._continuous_drift_calc(s1, s1, bins=10)
        assert drift_identical == 0 or drift_identical < 0.01, "Identical distributions should have drift close to 0"

        # Check different bin sizes
        drift_more_bins = instance_to_check.__class__._continuous_drift_calc(s1, s2, bins=20)
        assert isinstance(drift_more_bins, float)
        assert 0 <= drift_more_bins <= 1
