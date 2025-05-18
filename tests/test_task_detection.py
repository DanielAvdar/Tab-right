import pandas as pd
import pytest
from hypothesis import given, strategies as st

from tab_right.task_detection import TaskType, detect_task


# Keep original hypothesis tests for backward compatibility
@given(st.lists(st.integers(min_value=0, max_value=1), min_size=2, max_size=100))
def test_detect_task_binary(ints):
    # Only 0 and 1
    s = pd.Series(ints)
    if len(set(ints)) == 1:
        with pytest.raises(ValueError):
            detect_task(s)
    else:
        assert detect_task(s) == TaskType.BINARY


@given(st.lists(st.integers(min_value=0, max_value=9), min_size=2, max_size=100))
def test_detect_task_multiclass(ints):
    # 2 < n_unique <= 10
    s = pd.Series(ints)
    n_unique = len(set(ints))
    if n_unique == 1:
        with pytest.raises(ValueError):
            detect_task(s)
    elif n_unique == 2:
        assert detect_task(s) == TaskType.BINARY
    else:
        assert detect_task(s) == TaskType.CLASS


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=11, max_size=100))
def test_detect_task_regression(floats):
    # More than 10 unique values
    s = pd.Series(floats)
    if len(set(floats)) == 1:
        with pytest.raises(ValueError):
            detect_task(s)
    else:
        assert detect_task(s) == TaskType.REG


@given(st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), min_size=2, max_size=100))
def test_detect_task_categorical_class(strings):
    s = pd.Series(strings, dtype="category")
    n_unique = len(set(strings))
    if n_unique == 1:
        with pytest.raises(ValueError):
            detect_task(s)
    elif n_unique == 2:
        assert detect_task(s) == TaskType.BINARY
    else:
        assert detect_task(s) == TaskType.CLASS


def test_detect_task_many_integers():
    """Test detection of regression task for many unique integer values (>10)."""
    # Create a Series with 15 unique integer values (more than 10)
    s = pd.Series(list(range(15)))
    # This should trigger the 'else' branch (line 47) for TaskType.REG
    assert detect_task(s) == TaskType.REG


# New parameterized tests - more concise way to test similar scenarios
@pytest.mark.parametrize(
    "test_data, expected_type, description",
    [
        ([0, 1], TaskType.BINARY, "Basic binary integers"),
        (["yes", "no", "yes"], TaskType.BINARY, "Binary strings as categories"),
        ([0, 1, 2], TaskType.CLASS, "Simple multi-class"),
        (list(range(10)), TaskType.CLASS, "Upper boundary multi-class"),
        ([1.1, 2.2, 3.3, 4.4], TaskType.REG, "Float values for regression"),
        (list(range(15)), TaskType.REG, "Many integers for regression"),
    ],
)
def test_task_detection_parameterized(test_data, expected_type, description):
    """Parameterized test for task detection with various scenarios."""
    s = pd.Series(test_data)
    result = detect_task(s)
    assert result == expected_type, f"Failed for case: {description}"
