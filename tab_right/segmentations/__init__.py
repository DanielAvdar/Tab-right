"""Segmentation subpackage for tab-right: provides tools for segmenting tabular data for model diagnostics."""

from .calc_seg import SegmentationStats
from .double_seg import DecisionTreeDoubleSegmentation
from .find_seg import DecisionTreeSegmentation

__all__ = ["DecisionTreeSegmentation", "DecisionTreeDoubleSegmentation", "SegmentationStats"]
