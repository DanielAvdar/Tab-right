import pandas as pd
import matplotlib.pyplot as plt
from tab_right.plotting import plot_single_segmentation_mp

# Create a DataFrame with example R² values by segment
r2_segments = pd.DataFrame({
    'segment_id': [0, 1, 2, 3],
    'segment_name': ['Age < 30', '30 ≤ Age < 50', '50 ≤ Age < 65', 'Age ≥ 65'],
    'score': [0.82, 0.91, 0.76, 0.68]  # R² values (higher is better)
})

# Plot with lower_is_better=False for R²
plot_single_segmentation_mp(r2_segments, lower_is_better=False)
plt.title("R² by Age Segment")
plt.show()