import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tab_right.segmentations import DoubleSegmentationImp
from tab_right.plotting import DoubleSegmPlottingMp

# Create sample data with mixed feature types
np.random.seed(42)
n_samples = 500

# Generate categorical feature - product type
product_types = ['Basic', 'Standard', 'Premium', 'Enterprise']
product = np.random.choice(product_types, n_samples, p=[0.4, 0.3, 0.2, 0.1])

# Generate continuous feature - customer spending
spending = np.random.gamma(shape=5, scale=20, size=n_samples)

# Add variation by product type
spending[product == 'Premium'] *= 1.5
spending[product == 'Enterprise'] *= 2.0

# Simple model: customers return if they have premium products OR spend a lot
premium_mask = np.logical_or(product == 'Premium', product == 'Enterprise')
return_prob = 0.2 + 0.3 * premium_mask + 0.4 * (spending > np.percentile(spending, 70))
return_prob = np.clip(return_prob, 0.1, 0.9)

# Generate actual returns (target)
customer_return = np.random.binomial(1, return_prob)

# Simple prediction (missing some patterns)
pred_prob = 0.2 + 0.4 * (product == 'Enterprise') + 0.3 * (spending > np.percentile(spending, 80))
pred_prob = np.clip(pred_prob, 0.1, 0.9)
prediction = np.random.binomial(1, pred_prob)

# Create DataFrame
mixed_df = pd.DataFrame({
    'product': product,
    'spending': spending,
    'target': customer_return,
    'prediction': prediction
})

# Perform double segmentation
mixed_seg = DoubleSegmentationImp(
    df=mixed_df,
    label_col='target',
    prediction_col='prediction'
)

# Apply segmentation
mixed_results = mixed_seg(
    feature1_col='product',
    feature2_col='spending',
    score_metric=f1_score,
    bins_2=4  # 4 bins for spending
)

# Plot with higher is better for F1 score
mixed_plot = DoubleSegmPlottingMp(
    df=mixed_results,
    lower_is_better=False
)
fig = mixed_plot.plot_heatmap()
plt.title("F1 Score by Product Type and Spending")