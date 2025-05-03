from tab_right.base_architecture.seg_protocols_check import TestBaseSegmentationCalc

import pytest




class TestBaseSegmentationCalcImp(TestBaseSegmentationCalc):
    """Test class for double segmentation."""

    @pytest.fixture
    def instance_to_check(self) -> TestBaseSegmentationCalc:
        """Fixture to create an instance of the class."""
        # todo: create implementation of instance_to_check instance.