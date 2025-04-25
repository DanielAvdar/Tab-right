from enum import Enum
import pandas as pd

class TaskType(Enum):
    BINARY = "binary"
    CLASS = "class"
    REG = "reg"

def detect_task(y: pd.Series) -> "TaskType":
    unique = set(y.dropna().unique())
    n_classes = len(unique)
    if n_classes == 1:
        raise ValueError("Label column has only one unique value; cannot infer task.")
    # If float dtype, always regression
    if pd.api.types.is_float_dtype(y):
        return TaskType.REG
    if pd.api.types.is_categorical_dtype(y) or y.dtype == object:
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
