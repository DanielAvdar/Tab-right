import pandas as pd
import pytest

from tab_right.base_architecture.model_comparison_protocols import PredictionCalculationP

from ..base_protocols_check import CheckProtocols


class CheckPredictionCalculation(CheckProtocols):
    """Class for checking compliance of `PredictionCalculationP` protocol."""

    # Use the protocol type directly
    class_to_check = PredictionCalculationP

    def test_attributes(self, instance_to_check: PredictionCalculationP) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "df")
        assert hasattr(instance_to_check, "label_col")
        assert isinstance(instance_to_check.df, pd.DataFrame)
        assert isinstance(instance_to_check.label_col, str)
        assert instance_to_check.label_col in instance_to_check.df.columns

    def test_call_method(self, instance_to_check: PredictionCalculationP) -> None:
        """Test the __call__ method of the instance."""
        # Create test prediction data
        n_samples = len(instance_to_check.df)
        pred_data = [
            pd.Series(range(n_samples), index=instance_to_check.df.index, name="pred_0"),
            pd.Series(range(n_samples, 2 * n_samples), index=instance_to_check.df.index, name="pred_1"),
        ]

        # Test with default error function
        result = instance_to_check(pred_data)
        assert isinstance(result, pd.DataFrame)

        # Check that the result contains original DataFrame columns
        for col in instance_to_check.df.columns:
            assert col in result.columns

        # Check that error columns are present
        assert "pred_0_error" in result.columns
        assert "pred_1_error" in result.columns

        # Test with custom error function
        def custom_error(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
            return abs(y_true - y_pred)

        result_custom = instance_to_check(pred_data, error_func=custom_error)
        assert isinstance(result_custom, pd.DataFrame)
        assert "pred_0_error" in result_custom.columns
        assert "pred_1_error" in result_custom.columns

        # Test with single prediction
        single_pred = [pred_data[0]]
        result_single = instance_to_check(single_pred)
        assert isinstance(result_single, pd.DataFrame)
        assert "pred_0_error" in result_single.columns
        assert "pred_1_error" not in result_single.columns

    def test_error_calculations(self, instance_to_check: PredictionCalculationP) -> None:
        """Test that error calculations work correctly."""
        # Create test data where we know the expected errors
        n_samples = len(instance_to_check.df)
        label_values = instance_to_check.df[instance_to_check.label_col]

        # Create prediction that should have zero error
        pred_exact = pd.Series(label_values.values, index=instance_to_check.df.index, name="pred_exact")

        # Create prediction with constant offset
        pred_offset = pd.Series(label_values.values + 1, index=instance_to_check.df.index, name="pred_offset")

        pred_data = [pred_exact, pred_offset]

        # Test with default error function
        result = instance_to_check(pred_data)

        # Check that exact prediction has zero error (or very close to zero)
        exact_errors = result["pred_0_error"]
        assert (exact_errors < 1e-10).all() or (exact_errors == 0).all()

        # Check that offset prediction has consistent non-zero error
        offset_errors = result["pred_1_error"]
        assert (offset_errors > 0).all()