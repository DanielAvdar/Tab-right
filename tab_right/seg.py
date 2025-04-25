"""Segmentation statistics for tab-right package."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from tab_right.task_detection import TaskType, detect_task


@dataclass
class SegmentationStats:
    """Compute statistics for segments in a DataFrame based on features, labels, and predictions."""

    df: pd.DataFrame
    label_col: Union[str, List[str]]
    pred_col: str
    feature: str
    metric: Optional[Callable] = None

    def _prepare_segments(self, bins: int, category_limit: int) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.df.copy()
        is_categorical = df[self.feature].nunique() <= category_limit
        if is_categorical:
            df["_segment"] = df[self.feature]
        else:
            df["_segment"] = pd.qcut(df[self.feature], q=bins, duplicates="drop")
        segments = df["_segment"].unique()
        return df, segments

    def _probability_mode(self, df: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
        segment_scores = []
        for seg in segments:
            mask = df["_segment"] == seg
            y_t = df.loc[mask, self.label_col]
            # Mean probability vector per segment
            score = y_t.mean().to_dict()
            segment_scores.append(score)
        return pd.DataFrame({"segment": segments, "score": segment_scores})

    def _get_metric(self, y_true: pd.Series) -> Tuple[Callable, Optional[TaskType]]:
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

    def _compute_segment_scores(self, df: pd.DataFrame, segments: pd.Series, metric_func: Callable, task: TaskType) -> pd.DataFrame:
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

    def run(self, bins: int = 10, category_limit: int = 20) -> pd.DataFrame:
        """Run segmentation statistics computation.

        Args:
            bins (int): Number of bins for continuous features.
            category_limit (int): Max unique values for categorical treatment.

        Returns:
            pd.DataFrame: Segment statistics.

        """
        df, segments = self._prepare_segments(bins, category_limit)
        # If label_col is a list, treat as probabilities, just return mean per segment
        if isinstance(self.label_col, list):
            return self._probability_mode(df, segments)
        # Otherwise, assume label_col is a single column
        y_true = df[self.label_col]
        metric_func, task = self._get_metric(y_true)
        return self._compute_segment_scores(df, segments, metric_func, task)

    def check(self) -> None:
        """Check the validity of label or probability columns in the DataFrame.

        Raises:
            ValueError: If NaN or invalid probability sums are found.

        """
        if isinstance(self.label_col, list):
            # Check for NaN in any probability column
            if self.df[self.label_col].isnull().any().any():
                raise ValueError("Probability columns contain NaN values.")
            # Check if probabilities sum to 1 (row-wise)
            prob_sums = self.df[self.label_col].sum(axis=1)
            if not ((prob_sums - 1).abs() < 1e-6).all():
                raise ValueError("Probabilities in label columns do not sum to 1 for all rows.")
        else:
            if self.df[self.label_col].isnull().any():
                raise ValueError("Label column contains NaN values.")
