from tab_right.base_architecture.seg_protocols import FindSegmentation

import pytest



class FindSegmentationImp(FindSegmentation):
    """Test class for double segmentation."""

    @pytest.fixture
    def instance_to_check(self) -> FindSegmentation:
        """Fixture to create an instance of the class."""
        # todo: create implementation of instance_to_check instance.