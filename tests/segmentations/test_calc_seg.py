import pandas as pd
import pytest

from tab_right.base_architecture.seg_protocols_check import CheckBaseSegmentationCalc
from tab_right.segmentations.calc_seg import BaseSegmentationCalcImp


class TestBaseSegmentationCalcImp(CheckBaseSegmentationCalc):
    """Test class for double segmentation."""

    @pytest.fixture
    def instance_to_check(self) -> CheckBaseSegmentationCalc:
        """Fixture to create an instance of the class."""
        data = {"segment_id": [1, 1, 2, 2], "label": [0, 1, 0, 1], "prediction": [0.1, 0.9, 0.2, 0.8]}
        df = pd.DataFrame(data)
        gdf = df.groupby("segment_id")
        return BaseSegmentationCalcImp(gdf, "label", "prediction")
