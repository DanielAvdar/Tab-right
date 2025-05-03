import pandas as pd
import pytest

from tab_right.base_architecture.seg_protocols_check import CheckDoubleSegmentation
from tab_right.segmentations.double_seg import DoubleSegmentationImp
from tab_right.segmentations.find_seg import FindSegmentationImp


class TestDoubleSegmentationImp(CheckDoubleSegmentation):
    """Test class for double segmentation."""

    @pytest.fixture
    def instance_to_check(self) -> CheckDoubleSegmentation:
        """Fixture to create an instance of the class."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "label": [0, 1, 0, 1],
            "prediction": [0.1, 0.9, 0.2, 0.8],
        })
        segmentation_finder = FindSegmentationImp(df, "label", "prediction")
        return DoubleSegmentationImp(segmentation_finder)
