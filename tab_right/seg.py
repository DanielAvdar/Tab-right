"""Tab-right package init."""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score, balanced_accuracy_score
from typing import Optional, Callable


@dataclass
class SegmentationStats:
    df: pd.DataFrame
    label_col: str
    pred_col: str
    feature: str
    metric: Optional[Callable] = None  # Now expects a callable or None

    def _detect_task_and_metric(self, y_true) -> (str, Callable):
        unique = set(y_true.dropna().unique())
        n_classes = len(unique)
        # Special case: only one unique value
        if n_classes == 1:
            raise ValueError("Label column has only one unique value; cannot infer task.")
        if self.metric is not None:
            return self._infer_task_from_metric(self.metric), self.metric
        # If categorical dtype, always classification
        if pd.api.types.is_categorical_dtype(y_true) or y_true.dtype == object:
            if n_classes == 2:
                return 'binary', roc_auc_score
            else:
                return 'multiclass', balanced_accuracy_score
        # If numeric
        if n_classes == 2:
            return 'binary', roc_auc_score
        elif n_classes <= 10:
            return 'multiclass', balanced_accuracy_score
        else:
            return 'regression', r2_score

    def _infer_task_from_metric(self, metric: Callable) -> str:
        # Heuristic: check function name
        name = getattr(metric, '__name__', str(metric))
        if name in ['r2_score', 'mean_squared_error']:
            return 'regression'
        elif name in ['roc_auc_score', 'accuracy_score']:
            return 'binary'
        elif name in ['balanced_accuracy_score']:
            return 'multiclass'
        else:
            return 'unknown'

    def run(self, bins: int = 10, category_limit: int = 20):
        df = self.df.copy()
        is_categorical = df[self.feature].nunique() <= category_limit
        if is_categorical:
            df["_segment"] = df[self.feature]
        else:
            df["_segment"] = pd.qcut(df[self.feature], q=bins, duplicates="drop")
        segments = df["_segment"].unique()
        y_true = df[self.label_col]
        task, metric_func = self._detect_task_and_metric(y_true)
        segment_scores = []
        for seg in segments:
            mask = df["_segment"] == seg
            y_t = df.loc[mask, self.label_col]
            y_p = df.loc[mask, self.pred_col]
            if task == "multiclass":
                score = metric_func(y_t, y_p.round())
            else:
                score = metric_func(y_t, y_p)
            segment_scores.append(score)
        return pd.DataFrame({"segment": segments, "score": segment_scores})
