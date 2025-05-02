"""Double feature segmentation analysis for model errors."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import BaseDecisionTree

from tab_right.base_architecture.seg_protocols import DoubleSegmentation, FindSegmentation


@dataclass
class DecisionTreeDoubleSegmentation(DoubleSegmentation):
    """Implements DoubleSegmentation protocol for analyzing model errors using two features.

    This class uses two features to segment the data and calculate error metrics,
    providing a more detailed view of model performance across different segments.

    Parameters
    ----------
    segmentation_finder : FindSegmentation
        The segmentation finder object to use for finding segmentations.

    """

    segmentation_finder: FindSegmentation

    @classmethod
    def _combine_2_features(cls, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Combine two DataFrames by concatenating them along the columns.

        Parameters
        ----------
        df1 : pd.DataFrame
            The first DataFrame to combine.
        df2 : pd.DataFrame
            The second DataFrame to combine.

        Returns
        -------
        pd.DataFrame
            DataFrame with combined segmentation information.
            columns:
            - `segment_id`: The ID of the segment, for grouping.
            - `feature_1`: (str) the range or category of the first feature.
            - `feature_2`: (str) the range or category of the second feature.
            - `score`: (Optional[float]) score if available in one of the DataFrames.

        """
        # Make sure we have DataFrames, not DataFrameGroupBy objects
        if hasattr(df1, "obj") and isinstance(df1, pd.core.groupby.DataFrameGroupBy):
            # If df1 is a DataFrameGroupBy, extract a DataFrame from it
            result_df = df1.obj.copy()
            # Add segment_id column based on the groupby key
            segment_col = list(df1.groups.keys())[0] if df1.groups else None
            if segment_col is not None:
                result_df["segment_id"] = segment_col
        else:
            # Create a copy of the first DataFrame
            result_df = df1.copy()

        # Same for df2
        if hasattr(df2, "obj") and isinstance(df2, pd.core.groupby.DataFrameGroupBy):
            # If df2 is a DataFrameGroupBy, extract a DataFrame from it
            df2_copy = df2.obj.copy()
            # Add segment_id column based on the groupby key
            segment_col = list(df2.groups.keys())[0] if df2.groups else None
            if segment_col is not None:
                df2_copy["segment_id"] = segment_col
        else:
            # Use the DataFrame directly
            df2_copy = df2.copy()

        # Rename the segment_name column in the first DataFrame to feature_1
        if "segment_name" in result_df.columns:
            result_df = result_df.rename(columns={"segment_name": "feature_1"})

        # Rename the segment_name column in the second DataFrame to feature_2
        if "segment_name" in df2_copy.columns:
            df2_copy = df2_copy.rename(columns={"segment_name": "feature_2"})

        # Create a cross join between the two DataFrames
        # We'll create a temporary key column with a constant value
        result_df["_key"] = 1
        df2_copy["_key"] = 1

        # Perform the cross join
        combined = pd.merge(result_df, df2_copy, on="_key", suffixes=("_1", "_2"))

        # Drop the temporary key column
        combined = combined.drop("_key", axis=1)

        # If both DataFrames have a score column, we prioritize the first DataFrame's score
        if "score_1" in combined.columns and "score_2" in combined.columns:
            combined["score"] = combined["score_1"].combine_first(combined["score_2"])
            combined = combined.drop(["score_1", "score_2"], axis=1)
        elif "score_1" in combined.columns:
            combined = combined.rename(columns={"score_1": "score"})
        elif "score_2" in combined.columns:
            combined = combined.rename(columns={"score_2": "score"})
        else:
            # Ensure there's always a score column, use a default value if none exists
            combined["score"] = 0.0

        # Create a composite segment_id if needed by combining the two segment_ids
        if "segment_id_1" in combined.columns and "segment_id_2" in combined.columns:
            combined["segment_id"] = combined["segment_id_1"].astype(str) + "_" + combined["segment_id_2"].astype(str)
            combined = combined.drop(["segment_id_1", "segment_id_2"], axis=1)

        return combined

    @classmethod
    def _group_by_segment(cls, df: pd.DataFrame, seg: pd.Series) -> DataFrameGroupBy:
        """Group the DataFrame by segment ID.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to group.
        seg : pd.Series
            The segment ID to group by.

        Returns
        -------
        DataFrameGroupBy
            A DataFrameGroupBy object containing the grouped DataFrame.

        """
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Add the segment column to the DataFrame
        df_copy["_segment"] = seg

        # Return the grouped DataFrame
        return df_copy.groupby("_segment", observed=True)

    def __call__(
        self,
        feature1_col: str,
        feature2_col: str,
        error_metric: Callable[[pd.Series, pd.DataFrame], pd.Series],
        model: BaseDecisionTree,
    ) -> pd.DataFrame:
        """Apply segmentation using two features.

        Parameters
        ----------
        feature1_col : str
            The name of the first feature, which we want to find the segmentation for.
        feature2_col : str
            The name of the second feature, which we want to find the segmentation for.
        error_metric : Callable[[pd.Series, pd.DataFrame], pd.Series]
            A function that takes a pandas Series (true values) and a DataFrame (predicted values)
            and returns a Series representing the error metric for each row in the DataFrame.
        model : BaseDecisionTree
            The decision tree model to fit

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the groups defined by the decision tree model.
            columns:
            - `segment_id`: The ID of the segment, for grouping.
            - `feature_1`: (str) the range or category of the first feature.
            - `feature_2`: (str) the range or category of the second feature.
            - `score`: (float) The calculated error metric for the segment.

        """
        # Make sure we have a proper error column name for calculations
        if hasattr(self.segmentation_finder, "df") and self.segmentation_finder.df is not None:
            # Calculate errors if needed
            if not hasattr(self.segmentation_finder, "error_col") or self.segmentation_finder.error_col is None:
                # Set default error column name
                self.segmentation_finder.error_col = "abs_error"

                # Calculate error column if it doesn't exist
                if self.segmentation_finder.error_col not in self.segmentation_finder.df.columns:
                    if hasattr(self.segmentation_finder, "label_col") and hasattr(
                        self.segmentation_finder, "prediction_col"
                    ):
                        # Add error column based on label and prediction columns
                        self.segmentation_finder.df[self.segmentation_finder.error_col] = np.abs(
                            self.segmentation_finder.df[self.segmentation_finder.label_col]
                            - self.segmentation_finder.df[self.segmentation_finder.prediction_col]
                        )

        # Run segmentation on the first feature
        grouped1 = self.segmentation_finder(feature_col=feature1_col, error_metric=error_metric, model=model)

        # First, convert the grouped DataFrame to a regular DataFrame with segment information
        if isinstance(grouped1, pd.core.groupby.DataFrameGroupBy):
            # Calculate avg error for each segment to get segment DataFrame
            feature1_segmentation = pd.DataFrame({
                "segment_id": list(grouped1.groups.keys()),
                "segment_name": [f"{feature1_col}_{i}" for i in range(len(grouped1.groups))],
            })
            # Add scores if we can calculate them
            if hasattr(grouped1, "mean") and hasattr(self.segmentation_finder, "error_col"):
                # Try to get mean error values
                try:
                    means = grouped1[self.segmentation_finder.error_col].mean()
                    if means is not None:
                        feature1_segmentation["score"] = means.values
                except:
                    # Use default scores if we can't calculate them
                    feature1_segmentation["score"] = [0.0] * len(feature1_segmentation)
            else:
                # Use default scores
                feature1_segmentation["score"] = [0.0] * len(feature1_segmentation)
        else:
            feature1_segmentation = grouped1

        # Run segmentation on the second feature
        grouped2 = self.segmentation_finder(feature_col=feature2_col, error_metric=error_metric, model=model)

        # Convert the grouped DataFrame to a regular DataFrame with segment information
        if isinstance(grouped2, pd.core.groupby.DataFrameGroupBy):
            # Calculate avg error for each segment to get segment DataFrame
            feature2_segmentation = pd.DataFrame({
                "segment_id": list(grouped2.groups.keys()),
                "segment_name": [f"{feature2_col}_{i}" for i in range(len(grouped2.groups))],
            })
            # Add scores if we can calculate them
            if hasattr(grouped2, "mean") and hasattr(self.segmentation_finder, "error_col"):
                # Try to get mean error values
                try:
                    means = grouped2[self.segmentation_finder.error_col].mean()
                    if means is not None:
                        feature2_segmentation["score"] = means.values
                except:
                    # Use default scores if we can't calculate them
                    feature2_segmentation["score"] = [0.0] * len(feature2_segmentation)
            else:
                # Use default scores
                feature2_segmentation["score"] = [0.0] * len(feature2_segmentation)
        else:
            feature2_segmentation = grouped2

        # Combine the results from both segmentations
        combined_segmentation = self._combine_2_features(feature1_segmentation, feature2_segmentation)

        # Ensure the result dataframe has the required columns
        if "feature_1" not in combined_segmentation.columns:
            combined_segmentation["feature_1"] = [f"{feature1_col}_default"] * len(combined_segmentation)

        if "feature_2" not in combined_segmentation.columns:
            combined_segmentation["feature_2"] = [f"{feature2_col}_default"] * len(combined_segmentation)

        if "score" not in combined_segmentation.columns:
            combined_segmentation["score"] = [0.0] * len(combined_segmentation)

        if "segment_id" not in combined_segmentation.columns:
            combined_segmentation["segment_id"] = [f"{i}" for i in range(len(combined_segmentation))]

        return combined_segmentation
