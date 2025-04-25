"""Tab-right package init."""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score, balanced_accuracy_score
from typing import Optional, Callable, Union, List
from tab_right.task_detection import detect_task, TaskType


@dataclass
class SegmentationStats:
    df: pd.DataFrame
    label_col: Union[str, List[str]]
    pred_col: str
    feature: str
    metric: Optional[Callable] = None

    def _prepare_segments(self, bins: int, category_limit: int) -> (pd.DataFrame, pd.Series):
        df = self.df.copy()
        is_categorical = df[self.feature].nunique() <= category_limit
        if is_categorical:
            df["_segment"] = df[self.feature]
        else:
            df["_segment"] = pd.qcut(df[self.feature], q=bins, duplicates="drop")
        segments = df["_segment"].unique()
        return df, segments

    def _probability_mode(self, df, segments):
        segment_scores = []
        for seg in segments:
            mask = df["_segment"] == seg
            y_t = df.loc[mask, self.label_col]
            # Mean probability vector per segment
            score = y_t.mean().to_dict()
            segment_scores.append(score)
        return pd.DataFrame({"segment": segments, "score": segment_scores})

    def _get_metric(self, y_true):
        if self.metric is not None:
            return self.metric, None
        task = detect_task(y_true)
        if task == TaskType.BINARY:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score, task
        elif task == TaskType.CLASS:
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score, task
        else:
            from sklearn.metrics import r2_score
            return r2_score, task

    def _compute_segment_scores(self, df, segments, metric_func, task):
        segment_scores = []
        for seg in segments:
            mask = df["_segment"] == seg
            y_t = df.loc[mask, self.label_col]
            y_p = df.loc[mask, self.pred_col]
            if task == TaskType.CLASS:
                score = metric_func(y_t, y_p.round())
            else:
                score = metric_func(y_t, y_p)
            segment_scores.append(score)
        return pd.DataFrame({"segment": segments, "score": segment_scores})

    def run(self, bins: int = 10, category_limit: int = 20):
        df, segments = self._prepare_segments(bins, category_limit)
        # If label_col is a list, treat as probabilities, just return mean per segment
        if isinstance(self.label_col, list):
            return self._probability_mode(df, segments)
        # Otherwise, assume label_col is a single column
        y_true = df[self.label_col]
        metric_func, task = self._get_metric(y_true)
        return self._compute_segment_scores(df, segments, metric_func, task)
