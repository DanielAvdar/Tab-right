import pandas as pd

from tab_right.base_architecture.model_comparison_plot_protocols import ModelComparisonPlotP

from ..base_protocols_check import CheckProtocols


class CheckModelComparisonPlot(CheckProtocols):
    """Class for checking compliance of `ModelComparisonPlotP` protocol."""

    # Use the protocol type directly
    class_to_check = ModelComparisonPlotP

    def test_attributes(self, instance_to_check: ModelComparisonPlotP) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "comparison_calc")
        # The comparison_calc should implement PredictionCalculationP protocol
        assert hasattr(instance_to_check.comparison_calc, "df")
        assert hasattr(instance_to_check.comparison_calc, "label_col")
        assert callable(instance_to_check.comparison_calc)

    def test_plot_error_distribution(self, instance_to_check: ModelComparisonPlotP) -> None:
        """Test the plot_error_distribution method."""
        # Create test prediction data
        n_samples = len(instance_to_check.comparison_calc.df)
        [
            pd.Series(range(n_samples), index=instance_to_check.comparison_calc.df.index, name="pred_0"),
            pd.Series(range(n_samples, 2 * n_samples), index=instance_to_check.comparison_calc.df.index, name="pred_1"),
        ]

        # Test method exists and is callable
        assert hasattr(instance_to_check, "plot_error_distribution")
        assert callable(instance_to_check.plot_error_distribution)

        # Test that method can be called with default parameters
        # Note: We're not testing the actual plotting since these are protocols
        # The implementation would be tested separately

    def test_plot_pairwise_comparison(self, instance_to_check: ModelComparisonPlotP) -> None:
        """Test the plot_pairwise_comparison method."""
        # Test method exists and is callable
        assert hasattr(instance_to_check, "plot_pairwise_comparison")
        assert callable(instance_to_check.plot_pairwise_comparison)

    def test_plot_model_performance_summary(self, instance_to_check: ModelComparisonPlotP) -> None:
        """Test the plot_model_performance_summary method."""
        # Test method exists and is callable
        assert hasattr(instance_to_check, "plot_model_performance_summary")
        assert callable(instance_to_check.plot_model_performance_summary)
