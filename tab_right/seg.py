"""Tab-right package init."""

from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score


@dataclass
class SegmentationStats:
    df: pd.DataFrame
    label_col: str
    pred_col: str
    feature: str  # New parameter: the feature to segment by

    def run(self, bins: int = 10, category_limit: int = 20) -> pd.DataFrame:
        """
        Returns a DataFrame with segmentation statistics for the chosen feature:
        - count
        - mean label
        - mean prediction
        - std of prediction
        - accuracy (if label is binary)
        - error (MSE or accuracy, depending on task)
        """
        df = self.df.copy()
        is_categorical = df[self.feature].nunique() <= category_limit
        if is_categorical:
            df["_segment"] = df[self.feature]
        else:
            df["_segment"] = pd.qcut(df[self.feature], q=bins, duplicates="drop")
        grouped = df.groupby("_segment")
        result = grouped.agg(
            count=(self.label_col, "count"),
            mean_label=(self.label_col, "mean"),
            mean_pred=(self.pred_col, "mean"),
            std_pred=(self.pred_col, "std"),
        ).reset_index()
        y_true = df[self.label_col]
        if set(y_true.dropna().unique()).issubset({0, 1}):
            # Binary classification: use accuracy
            result["accuracy"] = grouped.apply(lambda g: accuracy_score(g[self.label_col], g[self.pred_col].round())).values
        else:
            # Regression: use MSE
            result["mse"] = grouped.apply(lambda g: mean_squared_error(g[self.label_col], g[self.pred_col])).values
        return result

    def plot(self, bins: int = 10, category_limit: int = 20, backend: str = "plotly", **kwargs):
        """
        Plot segmentation strength for the chosen feature.
        - For categorical features: bar plot of mean label/prediction per category.
        - For continuous features: bin feature, plot mean label/prediction per bin.
        """
        import importlib
        if importlib.util.find_spec("plotly") is None:
            raise ImportError("plotly is required for plotting. Install with `pip install tab-right[plotly]`.")
        import plotly.express as px
        df = self.df.copy()
        is_categorical = df[self.feature].nunique() <= category_limit
        if is_categorical:
            df["_segment"] = df[self.feature]
        else:
            df["_segment"] = pd.qcut(df[self.feature], q=bins, duplicates="drop")
        stats = df.groupby("_segment").agg(
            mean_label=(self.label_col, "mean"),
            mean_pred=(self.pred_col, "mean"),
        ).reset_index()
        fig = px.bar(stats, x="_segment", y=["mean_label", "mean_pred"], barmode="group", title=f"Segmentation by {self.feature}")
        fig.show()
