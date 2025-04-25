"""Tab-right package init."""

import pandas as pd


class SegmentationStats:
    def __init__(self, df: pd.DataFrame, label_col: str, pred_col: str):
        self.df = df
        self.label_col = label_col
        self.pred_col = pred_col

    def run(self) -> pd.DataFrame:
        """Returns a DataFrame with segmentation statistics:
        - count
        - mean label
        - mean prediction
        - std of prediction
        - accuracy (if label is binary).
        """
        result = (
            self.df.groupby(self.pred_col)
            .agg(
                count=(self.label_col, "count"),
                mean_label=(self.label_col, "mean"),
                mean_pred=(self.pred_col, "mean"),
                std_pred=(self.pred_col, "std"),
            )
            .reset_index()
        )
        # If label is binary, add accuracy
        if set(self.df[self.label_col].dropna().unique()).issubset({0, 1}):
            result["accuracy"] = (
                self.df.groupby(self.pred_col)
                .apply(lambda g: (g[self.label_col] == g[self.pred_col].round()).mean())
                .values
            )
        return result
