import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tab_right.segmentations import DoubleSegmentationImp
from tab_right.plotting import DoubleSegmPlottingMp

# Create sample categorical data
np.random.seed(42)
n = 800

# Generate categorical features with non-uniform distributions
category1 = np.random.choice(
    ['A', 'B', 'C', 'D'],
    n,
    p=[0.4, 0.3, 0.2, 0.1]  # Different probabilities for each category
)
category2 = np.random.choice(
    ['X', 'Y', 'Z'],
    n,
    p=[0.5, 0.3, 0.2]
)

# Generate target with different patterns for combinations
target = np.zeros(n, dtype=int)

# Add different effects for different combinations
target[(category1 == 'A') & (category2 == 'X')] = 1
target[(category1 == 'B') & (category2 == 'Y')] = 1
target[(category1 == 'C') & (category2 == 'Z')] = 1
# Special case with stronger effect
target[(category1 == 'D') & (category2 == 'Z')] = np.random.binomial(1, 0.8, np.sum((category1 == 'D') & (category2 == 'Z')))

# Add some noise
noise_mask = np.random.choice([True, False], n, p=[0.1, 0.9])
target[noise_mask] = 1 - target[noise_mask]

# Simple prediction without capturing all patterns
prediction = np.zeros(n, dtype=int)
prediction[category1 == 'A'] = 1
prediction[category2 == 'Z'] = 1

# Create DataFrame
cat_df = pd.DataFrame({
    'category1': category1,
    'category2': category2,
    'target': target,
    'prediction': prediction
})

# Perform double segmentation
cat_seg = DoubleSegmentationImp(
    df=cat_df,
    label_col='target',
    prediction_col='prediction'
)

# Apply segmentation (no bins needed for categorical features)
cat_results = cat_seg(
    feature1_col='category1',
    feature2_col='category2',
    score_metric=accuracy_score
)

# Plot with higher is better for accuracy
cat_plot = DoubleSegmPlottingMp(
    df=cat_results,
    lower_is_better=False
)
fig = cat_plot.plot_heatmap()
plt.title("Accuracy by Category Segments")