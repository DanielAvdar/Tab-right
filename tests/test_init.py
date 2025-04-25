import pytest
import pandas as pd
from tab_right.seg import SegmentationStats

import pytest

@pytest.mark.parametrize("df", [
    pd.DataFrame({
        'feature': ['a', 'a', 'b', 'b'],
        'class_0': [0.7, 0.6, 0.2, 0.1],
        'class_1': [0.3, 0.4, 0.8, 0.9],
    }),
    pd.DataFrame({
        'feature': ['x', 'x', 'y', 'y'],
        'class_0': [0.5, 0.5, 0.3, 0.3],
        'class_1': [0.5, 0.5, 0.7, 0.7],
    }),
])
def test_probability_mode_basic(df):
    seg = SegmentationStats(df, label_col=['class_0', 'class_1'], pred_col=None, feature='feature')
    result = seg.run()
    assert set(result['segment']) == set(df['feature'].unique())
    for score in result['score']:
        assert isinstance(score, dict)
        assert abs(sum(score.values()) - 1) < 1e-6

# Test check: NaN in probability columns
def test_check_nan_prob():
    df = pd.DataFrame({
        'feature': ['a', 'b'],
        'class_0': [0.5, None],
        'class_1': [0.5, 1.0],
    })
    seg = SegmentationStats(df, label_col=['class_0', 'class_1'], pred_col=None, feature='feature')
    with pytest.raises(ValueError, match='NaN'):
        seg.check()

# Test check: probabilities do not sum to 1
def test_check_prob_sum():
    df = pd.DataFrame({
        'feature': ['a', 'b'],
        'class_0': [0.6, 0.2],
        'class_1': [0.3, 0.7],
    })
    seg = SegmentationStats(df, label_col=['class_0', 'class_1'], pred_col=None, feature='feature')
    with pytest.raises(ValueError, match='sum to 1'):
        seg.check()

# Test check: valid probabilities
def test_check_valid_prob():
    df = pd.DataFrame({
        'feature': ['a', 'b'],
        'class_0': [0.4, 0.2],
        'class_1': [0.6, 0.8],
    })
    seg = SegmentationStats(df, label_col=['class_0', 'class_1'], pred_col=None, feature='feature')
    seg.check()
