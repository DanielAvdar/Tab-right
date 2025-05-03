import pandas as pd
import pytest

from tab_right.base_architecture.seg_protocols_check import CheckFindSegmentation
from tab_right.segmentations.find_seg import FindSegmentationImp


class TestFindSegmentationImp(CheckFindSegmentation):
    """Test class for double segmentation."""

    @pytest.fixture
    def instance_to_check(self):
        df = pd.DataFrame({"feature": [1, 2, 3, 4], "label": [0, 1, 0, 1], "prediction": [0.1, 0.9, 0.2, 0.8]})
        return FindSegmentationImp(df, "label", "prediction")
