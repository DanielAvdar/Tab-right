import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tab_right.plotting import plot_single_segmentation_mp

# Create a simple results DataFrame with segments
segments = pd.DataFrame({
    'segment_id': [0, 1, 2],
    'segment_name': ['Age < 30', '30 ≤ Age < 50', 'Age ≥ 50'],
    'score': [0.85, 0.92, 0.77]
})

# Plot the segmentation results using matplotlib
plot_single_segmentation_mp(segments)
plt.show()