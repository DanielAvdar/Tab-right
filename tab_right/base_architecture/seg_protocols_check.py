import abc

import pytest
from sklearn.tree import DecisionTreeClassifier

from .seg_protocols import DoubleSegmentation, FindSegmentation


class TestFindSegmentation:
    class_to_check = FindSegmentation

    @abc.abstractmethod
    @pytest.fixture
    def instance_to_check(self) -> FindSegmentation:
        """Fixture to create an instance of the class."""

    def test_call(self, instance_to_check):
        model = DecisionTreeClassifier()
        result = instance_to_check("feature", lambda y, p: abs(y - p.mean(axis=1)), model)
        assert "segment_id" in result.columns
        assert "segment_name" in result.columns
        assert "score" in result.columns

    def test_protocol_followed(self, instance_to_check):
        """Test if the protocol is followed correctly."""
        # Check if the class has the required attributes
        assert isinstance(instance_to_check, self.class_to_check)
        # check if it is dataclass
        assert hasattr(instance_to_check, "__dataclass_fields__")


class TestBaseSegmentationCalc:
    @abc.abstractmethod
    @pytest.fixture
    def instance_to_check(self) -> FindSegmentation:
        """Fixture to create an instance of the class."""

    def test_call(self, instance_to_check):
        result = instance_to_check(lambda y, p: abs(y - p).mean())
        assert "segment_id" in result.columns
        assert "score" in result.columns


class TestDoubleSegmentation:
    @abc.abstractmethod
    @pytest.fixture
    def instance_to_check(self) -> DoubleSegmentation:
        """Fixture to create an instance of the class."""

    def test_call(self, instance_to_check):
        model = DecisionTreeClassifier()
        result = instance_to_check("feature1", "feature2", lambda y, p: abs(y - p.mean(axis=1)), model)
        assert "segment_id" in result.columns
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "score" in result.columns
