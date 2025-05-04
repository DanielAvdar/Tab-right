"""Module for handling various segmentation algorithms and calculations.

This package provides functionality for calculating, finding, and analyzing segments
in data, including single and double segmentation approaches.
"""

from .calc_seg import SegmentationCalc
from .double_seg import DoubleSegmentationImp

__all__ = [
    "SegmentationCalc",
    "DoubleSegmentationImp",
]
