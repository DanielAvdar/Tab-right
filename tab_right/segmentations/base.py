from dataclasses import dataclass
from typing import Callable, List, Union
import pandas as pd

@dataclass
class SegmentationStats:
    df: pd.DataFrame
    label_col: Union[str, List[str]]
    pred_col: str
    feature: str
    metric: Callable
    is_categorical: bool = False  # True for categorical, False for continuous

    def _prepare_segments(self, bins: int = 10) -> pd.Series:
        if self.is_categorical:
            return self.df[self.feature]
        return pd.qcut(self.df[self.feature], q=bins, duplicates="drop")

    def run(self, bins: int = 10) -> pd.DataFrame:
        segments = self._prepare_segments(bins)
        df = self.df.copy()
        df["_segment"] = segments
        if isinstance(self.label_col, list):
            # Vectorized probability mode
            prob_means = df.groupby("_segment")[self.label_col].mean()
            prob_means = prob_means.reset_index().rename(columns={"_segment": "segment"})
            prob_means["score"] = prob_means[self.label_col].apply(lambda row: row.to_dict(), axis=1)
            return prob_means[["segment", "score"]]
        # Vectorized metric application
        def score_func(group):
            return float(self.metric(group[self.label_col], group[self.pred_col]))
        scores = df.groupby("_segment").apply(score_func)
        return pd.DataFrame({"segment": scores.index, "score": scores.values})

    def check(self) -> None:
        if isinstance(self.label_col, list):
            if self.df[self.label_col].isnull().values.any():
                raise ValueError("Probability columns contain NaN values.")
            prob_sums = self.df[self.label_col].sum(axis=1)
            if not ((prob_sums - 1).abs() < 1e-6).all():
                raise ValueError("Probabilities in label columns do not sum to 1 for all rows.")
        else:
            if self.df[self.label_col].isnull().any():
                raise ValueError("Label column contains NaN values.")
