from dataclasses import dataclass
from .base import BaseSegmentationStats
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
from tab_right.task_detection import TaskType

@dataclass
class ContinuousSegmentationStats(BaseSegmentationStats):
    def _prepare_segments(self, bins: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.df.copy()
        df["_segment"] = pd.qcut(df[self.feature], q=bins, duplicates="drop")
        segments = pd.Series(df["_segment"].unique())
        return df, segments

    def run(self, bins: int = 10) -> pd.DataFrame:
        df, segments = self._prepare_segments(bins)
        if isinstance(self.label_col, list):
            return self._probability_mode(df, segments)
        return self._compute_segment_scores(df, segments)
