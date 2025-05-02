import numpy as np
from sklearn.tree import DecisionTreeRegressor

def error_metric(y_true, y_pred):
    """Default error metric for regression tests."""
    if hasattr(y_pred, 'iloc') and len(getattr(y_pred, 'columns', [])) >= 1:
        pred_values = y_pred.iloc[:, 0]
    else:
        pred_values = y_pred
    return np.abs(y_true - pred_values)


def create_decision_tree_model(max_depth=2):
    """Create a DecisionTreeRegressor with a given max_depth."""
    return DecisionTreeRegressor(max_depth=max_depth)
