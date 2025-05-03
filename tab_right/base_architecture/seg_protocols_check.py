import abc
from typing import Any

import pandas as pd
import pytest
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import DecisionTreeRegressor

from .seg_protocols import BaseSegmentationCalc, DoubleSegmentation, FindSegmentation


class CheckProtocols:
    @abc.abstractmethod
    @pytest.fixture
    def instance_to_check(self) -> Any:
        """Fixture to create an instance of the class."""

    def test_protocol_followed(self, instance_to_check):
        """Test if the protocol is followed correctly."""
        # check if it is dataclass
        assert hasattr(instance_to_check, "__dataclass_fields__")
        assert isinstance(instance_to_check, self.class_to_check)


class CheckFindSegmentation(CheckProtocols):
    class_to_check = FindSegmentation

    def test_attributes(self, instance_to_check):
        assert hasattr(instance_to_check, "df")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_call(self, instance_to_check):
        model = DecisionTreeRegressor()
        result = instance_to_check("feature", lambda y, p: abs(y - p.mean(axis=1)), model)
        assert "segment_id" in result.columns
        assert "segment_name" in result.columns
        assert "score" in result.columns

    def test_calc_error(self, instance_to_check):
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.DataFrame({"prob_0": [0.8, 0.2, 0.7, 0.3], "prob_1": [0.2, 0.8, 0.3, 0.7]})

        def metric(y, p):
            return abs(y - p["prob_1"])

        result = instance_to_check._calc_error(metric, y_true, y_pred)
        assert len(result) == len(y_true)

    def test_fit_model(self, instance_to_check):
        feature = pd.Series([1, 2, 3, 4])
        error = pd.Series([0.1, 0.2, 0.3, 0.4])
        model = DecisionTreeRegressor()
        fitted_model = instance_to_check._fit_model(model, feature, error)
        assert hasattr(fitted_model, "tree_")

    def test_extract_leaves(self, instance_to_check):
        model = DecisionTreeRegressor(max_depth=2)
        feature = pd.Series([1, 2, 3, 4])
        error = pd.Series([0.1, 0.2, 0.3, 0.4])
        model.fit(feature.values.reshape(-1, 1), error)
        leaves = instance_to_check._extract_leaves(model)
        assert "segment_id" in leaves.columns
        assert "segment_name" in leaves.columns


class CheckBaseSegmentationCalc(CheckProtocols):
    class_to_check = BaseSegmentationCalc

    def test_attributes(self, instance_to_check):
        assert hasattr(instance_to_check, "gdf")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_call(self, instance_to_check):
        result = instance_to_check(lambda y, p: abs(y - p).mean())
        assert "segment_id" in result.columns
        assert "score" in result.columns


class CheckDoubleSegmentation(CheckProtocols):
    class_to_check = DoubleSegmentation

    def test_attributes(self, instance_to_check):
        assert hasattr(instance_to_check, "segmentation_finder")
        assert isinstance(instance_to_check.segmentation_finder, FindSegmentation)

    def test_call(self, instance_to_check):
        model = DecisionTreeRegressor()
        result = instance_to_check("feature1", "feature2", lambda y, p: abs(y - p.mean(axis=1)), model)
        assert "segment_id" in result.columns
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "score" in result.columns

    def test_combine_2_features(self, instance_to_check):
        df1 = pd.DataFrame({"segment_id": [1, 2], "segment_name": ["A", "B"], "score": [0.1, 0.2]})
        df2 = pd.DataFrame({"segment_id": [1, 2], "segment_name": ["C", "D"], "score": [0.3, 0.4]})
        combined = instance_to_check._combine_2_features(df1, df2)
        assert "feature_1" in combined.columns
        assert "feature_2" in combined.columns

    def test_group_by_segment(self, instance_to_check):
        df = pd.DataFrame({"segment_id": [1, 1, 2, 2], "score": [0.1, 0.2, 0.3, 0.4]})
        seg = df["segment_id"]
        grouped = instance_to_check._group_by_segment(df, seg)
        assert isinstance(grouped, DataFrameGroupBy)
