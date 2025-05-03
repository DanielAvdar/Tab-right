import abc

import pandas as pd
import pytest
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import DecisionTreeRegressor

from .seg_protocols import DoubleSegmentation, FindSegmentation


class CheckProtocols:
    class_to_check = FindSegmentation

    @abc.abstractmethod
    @pytest.fixture
    def instance_to_check(self) -> FindSegmentation:
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

    def test_calc_error(self):
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.DataFrame({"prob_0": [0.8, 0.2, 0.7, 0.3], "prob_1": [0.2, 0.8, 0.3, 0.7]})

        def metric(y, p):
            return abs(y - p["prob_1"])
        result = FindSegmentation._calc_error(metric, y_true, y_pred)
        assert len(result) == len(y_true)

    def test_fit_model(self):
        feature = pd.Series([1, 2, 3, 4])
        error = pd.Series([0.1, 0.2, 0.3, 0.4])
        model = DecisionTreeRegressor()
        fitted_model = FindSegmentation._fit_model(model, feature, error)
        assert hasattr(fitted_model, "tree_")

    def test_extract_leaves(self):
        model = DecisionTreeRegressor(max_depth=2)
        feature = pd.Series([1, 2, 3, 4])
        error = pd.Series([0.1, 0.2, 0.3, 0.4])
        model.fit(feature.values.reshape(-1, 1), error)
        leaves = FindSegmentation._extract_leaves(model)
        assert "segment_id" in leaves.columns
        assert "segment_name" in leaves.columns


class CheckBaseSegmentationCalc(CheckProtocols):
    def test_attributes(self, instance_to_check):
        assert hasattr(instance_to_check, "df")
        assert hasattr(instance_to_check, "label_col")
        assert hasattr(instance_to_check, "prediction_col")

    def test_call(self, instance_to_check):
        result = instance_to_check(lambda y, p: abs(y - p).mean())
        assert "segment_id" in result.columns
        assert "score" in result.columns


class CheckDoubleSegmentation(CheckProtocols):

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

    def test_combine_2_features(self):
        df1 = pd.DataFrame({"segment_id": [1, 2], "segment_name": ["A", "B"], "score": [0.1, 0.2]})
        df2 = pd.DataFrame({"segment_id": [1, 2], "segment_name": ["C", "D"], "score": [0.3, 0.4]})
        combined = DoubleSegmentation._combine_2_features(df1, df2)
        assert "feature_1" in combined.columns
        assert "feature_2" in combined.columns

    def test_group_by_segment(self):
        df = pd.DataFrame({"segment_id": [1, 1, 2, 2], "score": [0.1, 0.2, 0.3, 0.4]})
        seg = df["segment_id"]
        grouped = DoubleSegmentation._group_by_segment(df, seg)
        assert isinstance(grouped, DataFrameGroupBy)
