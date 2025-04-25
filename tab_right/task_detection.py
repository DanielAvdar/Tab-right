"""Task detection utilities for tab-right package."""

from enum import Enum

import pandas as pd


class TaskType(Enum):
    """Enumeration of possible task types for model evaluation."""

    BINARY = "binary"
    CLASS = "class"
    REG = "reg"


def detect_task(y: pd.Series) -> "TaskType":
    """Detect the type of task (binary, class, regression) based on the label series y.

    Args:
        y (pd.Series): The label series to analyze.

    Returns:
        TaskType: The detected task type.

    Raises:
        ValueError: If the label column has only one unique value and the task cannot be inferred.

    """
    unique = set(y.dropna().unique())
    n_classes = len(unique)
    if n_classes == 1:
        raise ValueError("Label column has only one unique value; cannot infer task.")
    # If float dtype, always regression
    if pd.api.types.is_float_dtype(y):
        return TaskType.REG
    # Deprecated: Use isinstance instead of is_categorical_dtype
    if isinstance(y.dtype, pd.CategoricalDtype) or y.dtype == object:
        if n_classes == 2:
            return TaskType.BINARY
        else:
            return TaskType.CLASS
    if n_classes == 2:
        return TaskType.BINARY
    elif n_classes <= 10:
        return TaskType.CLASS
    else:
        return TaskType.REG
