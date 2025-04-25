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
        if self.metric is not None:
            return self._infer_task_from_metric(self.metric), self.metric
        # Auto-detect
        if n_classes == 2 and unique.issubset({0, 1}):
            return 'binary', roc_auc_score  # Default: AUC for binary
        elif n_classes > 2 and all(isinstance(x, (int, float)) for x in unique):
            return 'multiclass', balanced_accuracy_score  # Suggest balanced accuracy for multiclass
        else:
            return 'regression', r2_score  # Default: R2 for regression

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

    def run(self, bins: int = 10, category_limit: int = 20) -> pd.DataFrame:
        """Returns a DataFrame with segmentation statistics for the chosen feature:
        - count
        - mean label
        - mean prediction
        - std of prediction
        - accuracy (if label is binary)
        - error (MSE or accuracy, depending on task).
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
        task, metric_func = self._detect_task_and_metric(y_true)
        if task == 'binary':
            try:
                result['auc'] = grouped.apply(lambda g: metric_func(g[self.label_col], g[self.pred_col])).values
            except Exception:
                result['auc'] = None
        elif task == 'multiclass':
            result['balanced_accuracy'] = grouped.apply(lambda g: metric_func(g[self.label_col], g[self.pred_col].round())).values
        else:
            result['r2'] = grouped.apply(lambda g: metric_func(g[self.label_col], g[self.pred_col])).values
        return result
