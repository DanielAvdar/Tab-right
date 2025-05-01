"""Tests for protocol implementations in tab-right."""

from typing import List, cast

import pandas as pd
from pandas.api.typing import DataFrameGroupBy
from sklearn.metrics import accuracy_score

from tab_right.base.seg_protocols import (
    BaseSegmentationCalc,
    PredictionSegmentationCalc,
    SegmentationCalc,
)
from tab_right.segmentations.base import SegmentationStats


class TestProtocols:
    """Test class for protocol implementations."""

    def test_segmentation_stats_implements_base_protocol(self):
        """Test that SegmentationStats implements the BaseSegmentationCalc protocol."""
        # Create simple test data
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
            "pred": [0, 1, 0, 1],
        })

        # Create SegmentationStats instance
        seg = SegmentationStats(
            df=df,
            label_col="label",
            prediction_col="pred",
            feature="feature",
            metric=accuracy_score,
            is_categorical=False,
        )

        # Verify it implements the BaseSegmentationCalc protocol
        assert isinstance(seg, BaseSegmentationCalc)

        # Call the instance to verify it works as expected
        result = seg()
        assert "segment" in result.columns
        assert "score" in result.columns

    def test_segmentation_stats_implements_prediction_protocol(self):
        """Test that SegmentationStats implements the PredictionSegmentationCalc protocol."""
        # Create simple test data with clear groups to avoid empty groups
        df = pd.DataFrame({
            "feature": [1, 1, 2, 2],  # Make sure we have at least 2 samples per group
            "label": [0, 1, 0, 1],
            "pred": [0, 1, 0, 1],
        })

        # Create SegmentationStats instance
        seg = SegmentationStats(
            df=df,
            label_col="label",
            prediction_col="pred",
            feature="feature",
            metric=accuracy_score,
            is_categorical=True,  # Use categorical to ensure exact grouping
        )

        # Verify it implements the PredictionSegmentationCalc protocol
        assert isinstance(seg, PredictionSegmentationCalc)

        # For backward compatibility, also check SegmentationCalc
        assert isinstance(seg, SegmentationCalc)

        # Call the instance to verify it works with a different metric
        result = seg(metric=accuracy_score)  # Use accuracy which works with binary data
        assert "segment" in result.columns
        assert "score" in result.columns

    def test_probability_mode_implements_base_protocol(self):
        """Test that probability mode implements BaseSegmentationCalc protocol."""
        # Create test data with probability columns
        df = pd.DataFrame({
            "feature": ["a", "a", "b", "b"],
            "class_0": [0.7, 0.6, 0.2, 0.1],
            "class_1": [0.3, 0.4, 0.8, 0.9],
        })

        # Create a custom class that implements the ProbabilitySegmentationCalc protocol
        class ProbabilitySegmentation:
            def __init__(self, df: pd.DataFrame, label_col: List[str], feature: str):
                self.df = df
                self.label_col = label_col
                self.feature = feature
                self._gdf = df.groupby(feature)

            @property
            def gdf(self) -> DataFrameGroupBy:
                return self._gdf

            @gdf.setter
            def gdf(self, value: DataFrameGroupBy) -> None:
                self._gdf = value

            @property
            def probability_cols(self) -> List[str]:
                return self.label_col

            def __call__(self, metric):
                prob_means = self.gdf[self.label_col].mean()
                prob_means = prob_means.reset_index().rename(columns={self.feature: "segment"})
                prob_means["score"] = prob_means[self.label_col].apply(lambda row: row.to_dict(), axis=1)
                return prob_means[["segment", "score"]]

        # Create an instance
        prob_seg = ProbabilitySegmentation(
            df=df,
            label_col=["class_0", "class_1"],
            feature="feature",
        )

        # Verify it implements the BaseSegmentationCalc protocol
        assert isinstance(prob_seg, BaseSegmentationCalc)

        # Call the instance
        result = prob_seg(metric=None)  # Metric not used in this implementation
        assert "segment" in result.columns
        assert "score" in result.columns

    def test_probability_implementation_with_segmentation_stats(self):
        """Test that SegmentationStats can work with probability columns."""
        # Create test data with probability columns
        df = pd.DataFrame({
            "feature": ["a", "a", "b", "b"],
            "class_0": [0.7, 0.6, 0.2, 0.1],
            "class_1": [0.3, 0.4, 0.8, 0.9],
        })

        # Create SegmentationStats instance with probability columns
        seg = SegmentationStats(
            df=df,
            label_col=["class_0", "class_1"],
            prediction_col=None,  # No prediction col needed
            feature="feature",
            metric=accuracy_score,  # Not used in probability mode
            is_categorical=True,
        )

        # Call the instance
        result = seg()
        assert "segment" in result.columns
        assert "score" in result.columns

        # Verify the scores contain dictionaries with probability values
        for score in result["score"]:
            assert isinstance(score, dict)
            assert "class_0" in score and "class_1" in score
            assert abs(score["class_0"] + score["class_1"] - 1.0) < 1e-6

    def test_explicit_protocol_adherence(self):
        """Test explicit protocol adherence by creating protocol objects."""
        # Create test data
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
            "pred": [0, 1, 0, 1],
        })

        # Create SegmentationStats instance
        seg = SegmentationStats(
            df=df,
            label_col="label",
            prediction_col="pred",
            feature="feature",
            metric=accuracy_score,
            is_categorical=False,
        )

        # Use the protocols as type annotations to verify static type checking
        base_protocol: BaseSegmentationCalc = cast(BaseSegmentationCalc, seg)
        pred_protocol: PredictionSegmentationCalc = cast(PredictionSegmentationCalc, seg)

        # Use the protocol objects
        assert base_protocol.label_col == "label"
        assert pred_protocol.prediction_col == "pred"

        # Verify they work as expected
        base_result = base_protocol(metric=accuracy_score)
        pred_result = pred_protocol(metric=accuracy_score)

        assert base_result.equals(pred_result)
