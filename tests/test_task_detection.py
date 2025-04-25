import pytest
from hypothesis import given, strategies as st
import pandas as pd
from tab_right.task_detection import detect_task, TaskType

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

@given(st.lists(st.sampled_from(['a', 'b', 'c', 'd', 'e']), min_size=2, max_size=100))
def test_detect_task_categorical_class(strings):
    s = pd.Series(strings, dtype='category')
    n_unique = len(set(strings))
    if n_unique == 1:
        with pytest.raises(ValueError):
            detect_task(s)
    elif n_unique == 2:
        assert detect_task(s) == TaskType.BINARY
    else:
        assert detect_task(s) == TaskType.CLASS
