from tab_right.base_architecture.seg_protocols_check import TestDoubleSegmentation

import pytest



class TestDoubleSegmentationImp(TestDoubleSegmentation):
    """Test class for double segmentation."""

    @pytest.fixture
    def instance_to_check(self) -> TestDoubleSegmentation:
        """Fixture to create an instance of the class."""
        # todo: create implementation of instance_to_check instance.