from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import BaseDecisionTree


@dataclass
class SegmentationCalc(Protocol):
    """Class schema for segmentation performance calculations."""

    gdf: DataFrameGroupBy
    label_col: str
    prediction_col: str

    def __call__(self, metric: Callable[[pd.Series, pd.Series], float]) -> pd.DataFrame:
        """Call method to apply the metric to each group in the DataFrameGroupBy object."""


@dataclass
class FindSegmentation2F(Protocol):
    """Class schema for segmentation performance calculations."""

    df: pd.DataFrame
    feature1_col: str
    feature2_col: str
    error_col: str

    def train_tree_model(self, model: BaseDecisionTree) -> BaseDecisionTree:
        """Train the decision tree model."""

    def __call__(self, model: BaseDecisionTree) -> DataFrameGroupBy:
        """Call method to apply the model to the DataFrame."""


# def find_segmentation_combinations(df: pd.DataFrame, features: list[str], error_col: str) -> list[FindSegmentation2F]:
#     """Find all combinations of two features in the DataFrame and return a list of FindSegmentation2F objects."""
find_segmentation_combinations = Callable[[pd.DataFrame, list[FindSegmentation2F], str], list[FindSegmentation2F]]
