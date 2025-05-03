"""Module for handling various segmentation algorithms and calculations.

This package provides functionality for calculating, finding, and analyzing segments
in data, including single and double segmentation approaches.
"""

from tab_right.segmentations.calc_seg import BaseSegmentationCalcImp
from tab_right.segmentations.double_seg import DoubleSegmentation
from tab_right.segmentations.find_seg import FindSegmentation

__all__ = ["BaseSegmentationCalcImp", "DoubleSegmentation", "FindSegmentation"]
