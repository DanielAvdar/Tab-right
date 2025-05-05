import numpy as np
import pandas as pd
import pytest

from tab_right.drift.drift_calculator import DriftCalculator
from tab_right.plotting.drift_plotter import DriftPlotter

from .drift_plot_check import CheckDriftPlot


@pytest.fixture
def instance_to_check():
    """Provides an instance of DriftPlotter for protocol testing."""
    df1 = pd.DataFrame({"a": np.random.rand(10), "b": np.random.choice(["x", "y"], 10)})
    df2 = pd.DataFrame({"a": np.random.rand(10), "b": np.random.choice(["x", "y", "z"], 10)})
    calculator = DriftCalculator(df1, df2)
    return DriftPlotter(calculator)


class TestDriftPlot(CheckDriftPlot):
    """Test class for `DriftPlotP` protocol compliance."""

    # todo: implement instance_to_check pytest fixture - DONE
