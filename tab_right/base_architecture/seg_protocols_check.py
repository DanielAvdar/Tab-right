import pytest
import abc
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .seg_protocols import FindSegmentation, BaseSegmentationCalc, DoubleSegmentation


class TestFindSegmentation:
    class_to_check = FindSegmentation

    @abc.abstractmethod
    @pytest.fixture
    def fs(self) -> FindSegmentation:
        """Fixture to create an instance of the class."""

    def test_call(self, fs):

        model = DecisionTreeClassifier()
        result = fs('feature', lambda y, p: abs(y - p.mean(axis=1)), model)
        assert 'segment_id' in result.columns
        assert 'segment_name' in result.columns
        assert 'score' in result.columns

    def test_protocol_followed(self, fs):
        """Test if the protocol is followed correctly."""
        # Check if the class has the required attributes
        assert isinstance(fs, self.class_to_check)
        # check if it is dataclass
        assert hasattr(fs, '__dataclass_fields__')


class TestBaseSegmentationCalc:

    @abc.abstractmethod
    @pytest.fixture
    def bsc(self) -> BaseSegmentationCalc:
        """Fixture to create an instance of the class."""

    def test_call(self, bsc):

        result = bsc( lambda y, p: abs(y - p).mean())
        assert 'segment_id' in result.columns
        assert 'score' in result.columns


class TestDoubleSegmentation:

    @abc.abstractmethod
    @pytest.fixture
    def ds(self) -> DoubleSegmentation:
        """Fixture to create an instance of the class."""

    def test_call(self, ds):

        model = DecisionTreeClassifier()
        result = ds('feature1', 'feature2', lambda y, p: abs(y - p.mean(axis=1)), model)
        assert 'segment_id' in result.columns
        assert 'feature_1' in result.columns
        assert 'feature_2' in result.columns
        assert 'score' in result.columns