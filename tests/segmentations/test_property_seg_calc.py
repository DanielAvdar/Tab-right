"""Property-based tests for the segmentation calculator."""

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import column, data_frames, indexes

from tab_right.segmentations.calc_seg import SegmentationCalc


# Helper function to create a valid metric for testing
def sample_metric(y_true, y_pred):
    """Simple metric that returns the mean of predictions."""
    return float(np.mean(y_pred))


@given(
    # Generate DataFrames with random values and segmentation IDs
    df=data_frames(
        columns=[
            column("label", elements=st.floats(min_value=0, max_value=1)),
            column("prediction", elements=st.floats(min_value=0, max_value=1)),
            column("segment_id", elements=st.integers(min_value=1, max_value=5)),
        ],
        index=indexes(elements=st.integers(min_value=0, max_value=100), min_size=5, max_size=50),
    ),
)
def test_segmentation_calc_properties(df):
    """Test that SegmentationCalc properties hold for arbitrary valid inputs."""
    # Skip empty dataframes
    if df.empty:
        return

    # Setup the segmentation
    gdf = df.groupby("segment_id")
    segment_names = {i: f"Segment {i}" for i in range(1, 6)}  # Create segment names for IDs 1-5

    # Create the calculator
    calc = SegmentationCalc(
        gdf=gdf,
        label_col="label",
        prediction_col="prediction",
        segment_names=segment_names,
    )

    # Run the calculation
    result = calc(sample_metric)

    # Property 1: Result should be a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    # Property 2: Result should have exactly the columns we expect
    assert set(result.columns) == {"segment_id", "name", "score"}, "Result should have correct columns"

    # Property 3: All segments in segment_names should appear in the result
    assert set(result["segment_id"]) == set(segment_names.keys()), "All segments should be included"

    # Property 4: Result length should match number of segments
    assert len(result) == len(segment_names), "Result length should match number of segments"

    # Property 5: Segment names should be correctly mapped
    for _, row in result.iterrows():
        assert row["name"] == f"Segment {row['segment_id']}", "Segment names should match mapping"


@given(
    # Generate edge case data: empty groups
    segment_ids=st.lists(
        st.integers(min_value=1, max_value=5),
        min_size=0,
        max_size=3,
        unique=True,
    )
)
def test_segmentation_calc_edge_cases(segment_ids):
    """Test that SegmentationCalc handles edge cases properly."""
    # Create a dataframe with potentially no data for some segments
    data = []
    for seg_id in segment_ids:
        data.append({"label": 0.5, "prediction": 0.7, "segment_id": seg_id})

    # Create DataFrame (might be empty)
    df = pd.DataFrame(data) if data else pd.DataFrame(columns=["label", "prediction", "segment_id"])

    # Handle the empty DataFrame case
    if df.empty:
        df = pd.DataFrame({"label": [], "prediction": [], "segment_id": []})

    # Setup the segmentation
    gdf = df.groupby("segment_id") if not df.empty else df.groupby(df.columns[0]) if len(df.columns) > 0 else None

    # Skip if we couldn't create a groupby object
    if gdf is None:
        return

    segment_names = {i: f"Segment {i}" for i in range(1, 6)}  # Create segment names for IDs 1-5

    # Create the calculator
    calc = SegmentationCalc(
        gdf=gdf,
        label_col="label",
        prediction_col="prediction",
        segment_names=segment_names,
    )

    # Run the calculation
    result = calc(sample_metric)

    # All segments should still be in the result
    assert len(result) == len(segment_names), "All segments should be in the result even with edge case data"

    # Segments with no data should have NaN scores
    empty_segments = set(segment_names.keys()) - set(segment_ids)
    for seg_id in empty_segments:
        seg_row = result[result["segment_id"] == seg_id]
        assert pd.isna(seg_row["score"].iloc[0]), f"Score for empty segment {seg_id} should be NaN"
