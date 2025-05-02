import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def sample_data():
    """Create sample data for segmentation/statistics/decision tree tests."""
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        "feature1": np.random.uniform(-1, 1, n_samples),
        "feature2": np.random.uniform(0, 10, n_samples),
        "feature3": np.random.normal(0, 1, n_samples),
    })
    # Regression/classification targets
    df["y_true"] = 2 * df["feature1"] + 0.5 * df["feature2"] + np.random.normal(0, 1, n_samples)
    df["y_pred"] = df["y_true"] + np.random.normal(0, 0.5, n_samples)
    # Categorical for classification
    df["category"] = np.random.choice(["A", "B", "C"], n_samples)
    # Probabilities for classification
    df["prob_class1"] = np.abs(np.sin(df["feature1"]))
    df["prob_class2"] = 1 - df["prob_class1"]
    return df
