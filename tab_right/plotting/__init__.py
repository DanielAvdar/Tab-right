"""tab_right.plotting: Plotting utilities for tab-right package."""

from ._matplotlib_backend import plot_drift_values as plot_drift_mp, plot_feature_drift_kde_mp as plot_feature_drift_mp
from ._plotly_backend import plot_drift_values as plot_drift, plot_feature_drift_kde as plot_feature_drift
from .plot_segmentations import (
    DoubleSegmPlotting as DoubleSegmPlotting,
    DoubleSegmPlottingMp as DoubleSegmPlottingMp,
    plot_single_segmentation as plot_single_segmentation,
    plot_single_segmentation_mp as plot_single_segmentation_mp,
)

# Explicit re-exports for external API
__all__ = [
    "plot_drift",
    "plot_drift_mp",
    "plot_feature_drift",
    "plot_feature_drift_mp",
    "DoubleSegmPlotting",
    "DoubleSegmPlottingMp",
    "plot_single_segmentation",
    "plot_single_segmentation_mp",
]
