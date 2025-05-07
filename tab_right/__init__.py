"""Tab-right package init."""

from importlib.metadata import version

from .drift import DriftCalculator as DriftCalculator
from .segmentations import DoubleSegmentationImp as DoubleSegmentationImp, SegmentationCalc as SegmentationCalc

__version__ = version("tab-right")
