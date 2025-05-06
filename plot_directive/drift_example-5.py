import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tab_right.drift import univariate
from tab_right.drift.drift_calculator import DriftCalculator
from tab_right.plotting.drift_plotter import DriftPlotter

# Generate data
np.random.seed(42)
df_ref = pd.DataFrame({
    'feat1': np.random.normal(0, 1, 500),
    'feat2': np.random.choice(['A', 'B', 'C'], 500),
})

df_cur = pd.DataFrame({
    'feat1': np.random.normal(0.5, 1.5, 500),
    'feat2': np.random.choice(['A', 'B', 'C'], 500, p=[0.5, 0.3, 0.2]),
})

# Using DriftCalculator with default metrics
calc = DriftCalculator(df_ref, df_cur)

# Create a plotter
plotter = DriftPlotter(calc)

# Plot the results
fig = plotter.plot_multiple()
plt.title('Drift Analysis with Default Metrics')
plt.tight_layout()
plt.show()