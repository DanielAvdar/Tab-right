import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create figure with 3 subplots for different drift levels
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Generate base reference data
np.random.seed(42)
ref_data = np.random.normal(0, 1, 500)

# Create titles and drift scores (pre-calculated for simplicity)
titles = ['Slight Drift', 'Moderate Drift', 'Severe Drift']
drift_scores = [0.241, 0.587, 1.934]  # Example scores

# Create datasets with increasing levels of drift
shifted_data = [
    np.random.normal(0.2, 1.1, 500),  # slight drift
    np.random.normal(0.5, 1.3, 500),  # moderate drift
    np.random.normal(2.0, 1.8, 500)   # severe drift
]

# Plot for each drift level
for i, current_data in enumerate(shifted_data):
    # Create histograms manually
    ax = axes[i]
    bins = 20
    hist_range = (min(min(ref_data), min(current_data)),
                 max(max(ref_data), max(current_data)))

    ref_hist, bin_edges = np.histogram(ref_data, bins=bins, range=hist_range, density=True)
    cur_hist, _ = np.histogram(current_data, bins=bins, range=hist_range, density=True)

    # Plot the histograms
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, ref_hist, 'b-', label='Reference')
    ax.plot(bin_centers, cur_hist, 'r-', label='Current')

    # Set title
    ax.set_title(titles[i])
    ax.text(0.5, 0.9, "Drift Score: " + str(drift_scores[i]),
            transform=ax.transAxes, ha='center')
    ax.legend()

plt.tight_layout()
plt.show()