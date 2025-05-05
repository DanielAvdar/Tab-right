import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tab_right.base_architecture.drift_plot_protocols import DriftPlotP

from ..base_protocols_check import CheckProtocols


class CheckDriftPlot(CheckProtocols):
    """Class for checking compliance of `DriftPlot` protocol."""

    # Use the protocol type directly
    class_to_check = DriftPlotP

    def test_attributes(self, instance_to_check: DriftPlotP) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "drift_calc")
        assert isinstance(instance_to_check.drift_calc, self.class_to_check)

    def test_plot_multiple(self, instance_to_check: DriftPlotP) -> None:
        """Test the plot_multiple method of the instance."""
        # Test with default parameters
        result = instance_to_check.plot_multiple()
        assert isinstance(result, (go.Figure, plt.Figure))

        # Test with specific parameters
        if len(instance_to_check.drift_calc.df1.columns) > 1:
            common_columns = set(instance_to_check.drift_calc.df1.columns) & set(
                instance_to_check.drift_calc.df2.columns
            )
            if len(common_columns) > 1:
                test_columns = list(common_columns)[:2]  # Just take 2 columns for testing
                result_subset = instance_to_check.plot_multiple(columns=test_columns)
                assert isinstance(result_subset, (go.Figure, plt.Figure))

        # Test with different sorting
        result_sorted = instance_to_check.plot_multiple(sort_by="feature", ascending=True)
        assert isinstance(result_sorted, (go.Figure, plt.Figure))

        # Test with top_n
        result_top = instance_to_check.plot_multiple(top_n=2)
        assert isinstance(result_top, (go.Figure, plt.Figure))

        # Test with threshold
        result_threshold = instance_to_check.plot_multiple(threshold=0.5)
        assert isinstance(result_threshold, (go.Figure, plt.Figure))

    def test_plot_single(self, instance_to_check: DriftPlotP) -> None:
        """Test the plot_single method of the instance."""
        # Get a common column to test
        common_columns = set(instance_to_check.drift_calc.df1.columns) & set(instance_to_check.drift_calc.df2.columns)
        if not common_columns:
            return  # Skip if no common columns

        test_column = list(common_columns)[0]

        # Test with default parameters
        result = instance_to_check.plot_single(column=test_column)
        assert isinstance(result, (go.Figure, plt.Figure))

        # Test with show_metrics=False
        result_no_metrics = instance_to_check.plot_single(column=test_column, show_metrics=False)
        assert isinstance(result_no_metrics, (go.Figure, plt.Figure))

        # Test with different figsize
        result_figsize = instance_to_check.plot_single(column=test_column, figsize=(8, 4))
        assert isinstance(result_figsize, (go.Figure, plt.Figure))

        # Test with different bins
        result_bins = instance_to_check.plot_single(column=test_column, bins=5)
        assert isinstance(result_bins, (go.Figure, plt.Figure))

    def test_get_distribution_plots(self, instance_to_check: DriftPlotP) -> None:
        """Test the get_distribution_plots method of the instance."""
        # Test with default parameters
        result = instance_to_check.get_distribution_plots()
        assert isinstance(result, dict)
        # Check that keys are feature names and values are figures
        for feature, figure in result.items():
            assert isinstance(feature, str)
            assert isinstance(figure, (go.Figure, plt.Figure))

        # Test with specific columns
        common_columns = set(instance_to_check.drift_calc.df1.columns) & set(instance_to_check.drift_calc.df2.columns)
        if len(common_columns) > 1:
            test_columns = list(common_columns)[:2]  # Just take 2 columns for testing
            result_subset = instance_to_check.get_distribution_plots(columns=test_columns)
            assert isinstance(result_subset, dict)
            assert set(result_subset.keys()) == set(test_columns)

        # Test with different bins
        result_bins = instance_to_check.get_distribution_plots(bins=5)
        assert isinstance(result_bins, dict)
        assert len(result_bins) > 0
