"""Module for checking segmentation protocols."""

import abc
from typing import Any

import pandas as pd
import pytest
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import DecisionTreeRegressor

from .seg_protocols import BaseSegmentationCalc, DoubleSegmentation, FindSegmentation


class CheckProtocols:
    """Base class for checking protocol compliance."""

    class_to_check = None  # type: ignore[assignment]

    @abc.abstractmethod
    @pytest.fixture
    def instance_to_check(self) -> Any:
        """Fixture to create an instance of the class."""

    def test_protocol_followed(self, instance_to_check: Any) -> None:
        """Test if the protocol is followed correctly."""
        # check if it is dataclass
        assert hasattr(instance_to_check, "__dataclass_fields__")
        assert isinstance(instance_to_check, self.class_to_check)


class CheckFindSegmentation(CheckProtocols):
    """Class for checking compliance of `FindSegmentation` protocol."""

    class_to_check = FindSegmentation

    def test_attributes(self, instance_to_check: Any) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "df")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_call(self, instance_to_check: Any) -> None:
        """Test the `__call__` method of the instance."""
        model = DecisionTreeRegressor()
        result = instance_to_check("feature", lambda y, p: abs(y - p.mean(axis=1)), model)
        assert "segment_id" in result.columns
        assert "segment_name" in result.columns
        assert "score" in result.columns

    def test_calc_error(self, instance_to_check: Any) -> None:
        """Test the error calculation method of the instance."""
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.DataFrame({"prob_0": [0.8, 0.2, 0.7, 0.3], "prob_1": [0.2, 0.8, 0.3, 0.7]})

        def metric(y, p):
            return abs(y - p["prob_1"])

        result = instance_to_check._calc_error(metric, y_true, y_pred)
        assert len(result) == len(y_true)

    def test_fit_model(self, instance_to_check):
        """Test the model fitting method of the instance."""
        feature = pd.Series([1, 2, 3, 4])
        error = pd.Series([0.1, 0.2, 0.3, 0.4])
        model = DecisionTreeRegressor()
        fitted_model = instance_to_check._fit_model(model, feature, error)
        assert hasattr(fitted_model, "tree_")

    def test_extract_leaves(self, instance_to_check):
        """Test the leaf extraction method of the instance."""
        model = DecisionTreeRegressor(max_depth=2)
        feature = pd.Series([1, 2, 3, 4])
        error = pd.Series([0.1, 0.2, 0.3, 0.4])
        model.fit(feature.values.reshape(-1, 1), error)
        leaves = instance_to_check._extract_leaves(model)
        assert "segment_id" in leaves.columns
        assert "segment_name" in leaves.columns


class CheckBaseSegmentationCalc(CheckProtocols):
    """Class for checking compliance of `BaseSegmentationCalc` protocol."""

    class_to_check = BaseSegmentationCalc

    def test_attributes(self, instance_to_check: Any) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "gdf")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_call(self, instance_to_check: Any) -> None:
        """Test the `__call__` method of the instance."""
        result = instance_to_check(lambda y, p: abs(y - p).mean())
        assert "segment_id" in result.columns
        assert "score" in result.columns


class CheckDoubleSegmentation(CheckProtocols):
    """Class for checking compliance of `DoubleSegmentation` protocol."""

    class_to_check = DoubleSegmentation

    def test_attributes(self, instance_to_check: Any) -> None:
        """Test attributes of the instance to ensure compliance."""
        assert hasattr(instance_to_check, "segmentation_finder")
        assert isinstance(instance_to_check.segmentation_finder, FindSegmentation)

    def test_call(self, instance_to_check: Any) -> None:
        """Test the `__call__` method of the instance."""
        model = DecisionTreeRegressor()
        result = instance_to_check("feature1", "feature2", lambda y, p: abs(y - p.mean(axis=1)), model)
        assert "segment_id" in result.columns
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "score" in result.columns

    def test_combine_2_features(self, instance_to_check):
        segment_id1 = list(range(6))
        segment_name1 = ["A", "B", "C", "D", "E", "F"]
        score1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        segment_id2 = list(range(6, 12))
        segment_name2 = ["G", "H", "I", "J", "K", "L"]
        score2 = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        df1 = pd.DataFrame({"segment_id": segment_id1, "feature_1": segment_name1, "score_1": score1})
        df2 = pd.DataFrame({"segment_id": segment_id2, "feature_2": segment_name2, "score_2": score2})
        combined = instance_to_check._combine_2_features(df1, df2)
        assert "feature_1" in combined.columns
        assert "feature_2" in combined.columns
        assert len(combined) == len(df1)
        assert len(combined) == len(df2)
        assert "score" in combined.columns
        assert combined["segment_id"].equals(pd.Series(range(6)))

    def test_group_by_segment(self, instance_to_check):
        df = pd.DataFrame({"segment_id": [1, 1, 2, 2], "score": [0.1, 0.2, 0.3, 0.4]})
        seg = df["segment_id"]
        grouped = instance_to_check._group_by_segment(df, seg)
        assert isinstance(grouped, DataFrameGroupBy)
