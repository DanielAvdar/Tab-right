

import pytest
import abc

from .seg_protocols import FindSegmentation, BaseSegmentationCalc, DoubleSegmentation


class TestFindSegmentation:

    @abc.abstractmethod
    @pytest.fixture
    def fs(self)-> FindSegmentation:
        """Fixture to create an instance of the class."""


class TestBaseSegmentationCalc:
    @abc.abstractmethod
    @pytest.fixture
    def bsc(self)-> BaseSegmentationCalc:
        """Fixture to create an instance of the class."""


class TestDoubleSegmentation:
    @abc.abstractmethod
    @pytest.fixture
    def ds(self)-> DoubleSegmentation:
        """Fixture to create an instance of the class."""