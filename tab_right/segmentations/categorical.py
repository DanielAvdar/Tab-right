from dataclasses import dataclass
from .base import BaseSegmentationStats
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
from tab_right.task_detection import TaskType

@dataclass
class CategoricalSegmentationStats(BaseSegmentationStats):
    def _prepare_segments(self, category_limit: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.df.copy()
        df["_segment"] = df[self.feature]
        segments = pd.Series(df["_segment"].unique())
        return df, segments

    def run(self, category_limit: int = 20) -> pd.DataFrame:
        df, segments = self._prepare_segments(category_limit)
        if isinstance(self.label_col, list):
            return self._probability_mode(df, segments)
        return self._compute_segment_scores(df, segments)
