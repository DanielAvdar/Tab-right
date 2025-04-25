"""Segmentation statistics for tab-right package."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from tab_right.task_detection import TaskType, detect_task


@dataclass
class CategoricalSegmentationStats:
    df: pd.DataFrame
    label_col: Union[str, List[str]]
    pred_col: str
    feature: str
    metric: Optional[Callable] = None

    def _prepare_segments(self, category_limit: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.df.copy()
        df["_segment"] = df[self.feature]
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

    def _compute_segment_scores(self, df: pd.DataFrame, segments: pd.Series, metric_func: Callable, task: TaskType) -> pd.DataFrame:
        def score_func(group: pd.DataFrame):
            y_t = group[self.label_col]
            y_p = group[self.pred_col]
            return float(metric_func(y_t, y_p))
        scores = df.groupby("_segment").apply(score_func)
        scores = scores.reset_index()
        scores.columns = ["segment", "score"]
        return scores

    def _get_metric(self, y_true: pd.Series) -> Tuple[Callable, Optional[TaskType]]:
        if self.metric is not None:
            return self.metric, None
        task = detect_task(y_true)
        if task == TaskType.BINARY:
            from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
            return roc_auc_score, task
        elif task == TaskType.CLASS:
            from sklearn.metrics import balanced_accuracy_score  # type: ignore[import-untyped]
            return balanced_accuracy_score, task
        else:
            from sklearn.metrics import r2_score  # type: ignore[import-untyped]
            return r2_score, task

    def run(self, category_limit: int = 20) -> pd.DataFrame:
        df, segments = self._prepare_segments(category_limit)
        if isinstance(self.label_col, list):
            return self._probability_mode(df, segments)
        y_true = df[self.label_col]
        metric_func, task = self._get_metric(y_true)
        return self._compute_segment_scores(df, segments, metric_func, task)

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


@dataclass
class ContinuousSegmentationStats:
    df: pd.DataFrame
    label_col: Union[str, List[str]]
    pred_col: str
    feature: str
    metric: Optional[Callable] = None

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

    def _compute_segment_scores(self, df: pd.DataFrame, segments: pd.Series, metric_func: Callable, task: TaskType) -> pd.DataFrame:
        def score_func(group: pd.DataFrame):
            y_t = group[self.label_col]
            y_p = group[self.pred_col]
            return float(metric_func(y_t, y_p))
        scores = df.groupby("_segment").apply(score_func)
        scores = scores.reset_index()
        scores.columns = ["segment", "score"]
        return scores

    def _get_metric(self, y_true: pd.Series) -> Tuple[Callable, Optional[TaskType]]:
        if self.metric is not None:
            return self.metric, None
        task = detect_task(y_true)
        if task == TaskType.BINARY:
            from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
            return roc_auc_score, task
        elif task == TaskType.CLASS:
            from sklearn.metrics import balanced_accuracy_score  # type: ignore[import-untyped]
            return balanced_accuracy_score, task
        else:
            from sklearn.metrics import r2_score  # type: ignore[import-untyped]
            return r2_score, task

    def run(self, bins: int = 10) -> pd.DataFrame:
        df, segments = self._prepare_segments(bins)
        if isinstance(self.label_col, list):
            return self._probability_mode(df, segments)
        y_true = df[self.label_col]
        metric_func, task = self._get_metric(y_true)
        return self._compute_segment_scores(df, segments, metric_func, task)

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
