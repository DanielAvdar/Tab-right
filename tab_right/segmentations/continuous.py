from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
from tab_right.task_detection import TaskType

@dataclass
class ContinuousSegmentationStats:
    df: pd.DataFrame
    label_col: Union[str, List[str]]
    pred_col: str
    feature: str
    metric: Callable
    task: TaskType

    def _prepare_segments(self, bins: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.df.copy()
        df["_segment"] = pd.qcut(df[self.feature], q=bins, duplicates="drop")
        segments = pd.Series(df["_segment"].unique())
        return df, segments

    def _probability_mode(self, df: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
        segment_scores = []
        for seg in segments:
            mask = df["_segment"] == seg
            y_t = df.loc[mask, self.label_col]
            score = y_t.mean().to_dict()
            segment_scores.append(score)
        return pd.DataFrame({"segment": segments, "score": segment_scores})

    def _compute_segment_scores(self, df: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
        def score_func(group: pd.DataFrame):
            y_t = group[self.label_col]
            y_p = group[self.pred_col]
            return float(self.metric(y_t, y_p))
        scores = df.groupby("_segment").apply(score_func)
        scores = scores.reset_index()
        scores.columns = ["segment", "score"]
        return scores

    def run(self, bins: int = 10) -> pd.DataFrame:
        df, segments = self._prepare_segments(bins)
        if isinstance(self.label_col, list):
            return self._probability_mode(df, segments)
        return self._compute_segment_scores(df, segments)

    def check(self) -> None:
        if isinstance(self.label_col, list):
            if self.df[self.label_col].isnull().any().any():
                raise ValueError("Probability columns contain NaN values.")
            prob_sums = self.df[self.label_col].sum(axis=1)
            if not ((prob_sums - 1).abs() < 1e-6).all():
                raise ValueError("Probabilities in label columns do not sum to 1 for all rows.")
        else:
            if self.df[self.label_col].isnull().any():
                raise ValueError("Label column contains NaN values.")
