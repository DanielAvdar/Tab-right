import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tab_right.segmentations import DoubleSegmentationImp
from tab_right.plotting import DoubleSegmPlottingMp

# Create sample data
np.random.seed(42)
n_samples = 500

# Generate features and target
feature1 = np.random.normal(0, 1, n_samples)
feature2 = np.random.normal(0, 1, n_samples)

# Target with interaction effect
target = 2 + feature1 + feature2 + 2 * (feature1 * feature2) + np.random.normal(0, 1, n_samples)

# Prediction missing the interaction term
prediction = 2 + feature1 + feature2 + np.random.normal(0, 1, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'target': target,
    'prediction': prediction
})

# Perform double segmentation
double_seg = DoubleSegmentationImp(
    df=df,
    label_col='target',
    prediction_col='prediction'
)

# Apply segmentation with 3 bins for each feature
result_df = double_seg(
    feature1_col='feature1',
    feature2_col='feature2',
    score_metric=mean_squared_error,
    bins_1=3,
    bins_2=3
)

# Visualize results with a heatmap
plotter = DoubleSegmPlottingMp(df=result_df)
fig = plotter.plot_heatmap()
plt.title("MSE by Feature1 and Feature2 Segments")