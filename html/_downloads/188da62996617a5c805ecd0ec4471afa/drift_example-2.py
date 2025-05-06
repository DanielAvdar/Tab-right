import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tab_right.drift.drift_calculator import DriftCalculator
from tab_right.plotting.drift_plotter import DriftPlotter

# Generate datasets with drift
np.random.seed(42)
df1 = pd.DataFrame({
    'numeric': np.random.normal(0, 1, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])
})

df2 = pd.DataFrame({
    'numeric': np.random.normal(1, 1.2, 120),
    'category': np.random.choice(['A', 'B', 'C'], 120, p=[0.2, 0.3, 0.5])
})

# Create calculator and plotter
drift_calc = DriftCalculator(df1, df2)
plotter = DriftPlotter(drift_calc)

# Plot numerical feature distribution comparison
fig_numeric = plotter.plot_single('numeric')
plt.tight_layout()
plt.show()