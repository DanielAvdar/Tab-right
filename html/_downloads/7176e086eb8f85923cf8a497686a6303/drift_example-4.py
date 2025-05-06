import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tab_right.drift import univariate
from tab_right.plotting import plot_drift_mp

# Generate datasets
np.random.seed(42)
df_ref = pd.DataFrame({
    'num_feature': np.random.normal(0, 1, 500),
    'cat_feature': np.random.choice(['A', 'B', 'C'], 500)
})

df_cur = pd.DataFrame({
    'num_feature': np.random.normal(0.3, 1.2, 500),
    'cat_feature': np.random.choice(['A', 'B', 'C'], 500, p=[0.2, 0.5, 0.3])
})

# Calculate drift across all features
result = univariate.detect_univariate_drift_df(df_ref, df_cur)

# Plot the results using matplotlib
fig = plot_drift_mp(result)
plt.tight_layout()
plt.show()