import numpy as np
import pandas as pd
import pytest

from tab_right.drift.drift_calculator import DriftCalculator

from .drift_check import CheckDriftCalc


@pytest.fixture
def instance_to_check():
    """Provides an instance of DriftCalculator for protocol testing."""
    df1 = pd.DataFrame({"a": np.random.rand(10), "b": np.random.choice(["x", "y"], 10)})
    df2 = pd.DataFrame({"a": np.random.rand(10), "b": np.random.choice(["x", "y", "z"], 10)})
    return DriftCalculator(df1, df2)


class TestDriftCalc(CheckDriftCalc):
    """Test class for `DriftCalcP` protocol compliance."""

    # todo: implement instance_to_check pytest fixture - DONE
