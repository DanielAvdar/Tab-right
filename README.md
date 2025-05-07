# tab-right

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tab-right)](https://pypi.org/project/tab-right/)
[![version](https://img.shields.io/pypi/v/tab-right)](https://pypi.org/project/tab-right/)
[![License](https://img.shields.io/:license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![OS](https://img.shields.io/badge/ubuntu-blue?logo=ubuntu)
![OS](https://img.shields.io/badge/win-blue?logo=windows)
![OS](https://img.shields.io/badge/mac-blue?logo=apple)
[![Tests](https://github.com/DanielAvdar/tab-right/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielAvdar/tab-right/actions/workflows/ci.yml)
[![Code Checks](https://github.com/DanielAvdar/tab-right/actions/workflows/code-checks.yml/badge.svg)](https://github.com/DanielAvdar/tab-right/actions/workflows/code-checks.yml)
[![codecov](https://codecov.io/gh/DanielAvdar/tab-right/graph/badge.svg?token=N0V9KANTG2)](https://codecov.io/gh/DanielAvdar/tab-right)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Last Commit](https://img.shields.io/github/last-commit/DanielAvdar/tab-right/main)

## Overview

**tab-right** is a Python package designed to simplify the analysis of tabular data for inference models—both machine learning and non-ML. The core philosophy is that most analyses, such as segmentation strength, drift analysis, and feature predictive value, can be performed using model predictions alone, without direct access to the model itself. This approach enables powerful, model-agnostic diagnostics and interpretability, making the package easy to implement and use.

## Key Features

- **Segmentation Analysis**: Analyze prediction strength across different data segments to uncover model biases and subgroup performance
- **Feature Analysis**: Assess feature predictive power and value to inference, using techniques like feature importance, partial dependence, and more
- **Drift Detection**: Perform drift analysis and monitor changes in data or prediction distributions over time
- **Rich Visualizations**: Generate comprehensive visualization reports for all analyses, supporting both interactive and static outputs
- **Model-Agnostic**: Focus on data and predictions, not model internals, for maximum flexibility and simplicity

## Installation

```bash
# Install from PyPI
pip install tab-right

# For development version
pip install git+https://github.com/DanielAvdar/tab-right.git
```

## Quick Start

Here's a simple example to get you started with tab-right:

```python
import pandas as pd
import numpy as np
from tab_right.segmentations import calc_seg

# Load your data
data = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.normal(0, 1, 1000),
    'predictions': np.random.uniform(0, 1, 1000)
})

# Perform segmentation analysis
segments = calc_seg(
    df=data,
    target_col='predictions',
    max_depth=3
)

# Print segmentation results
print(segments)
```

## Documentation

For detailed documentation and examples, visit our [documentation site](https://DanielAvdar.github.io/tab-right/).

The documentation includes:
- Comprehensive API reference
- In-depth tutorials
- Example notebooks
- Best practices guide

## Use Cases

- **Model Evaluation**: Compare model performance across different data segments
- **Model Monitoring**: Track model drift and data distribution changes over time
- **Feature Engineering**: Identify which features contribute most to predictions
- **Bias Detection**: Uncover potential biases in model predictions across subgroups

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use tab-right in a research paper, please cite it as:

```
@software{tab-right,
  author = {Avdar, Daniel},
  title = {tab-right: Model-Agnostic Analysis for Tabular Data},
  year = {2023},
  url = {https://github.com/DanielAvdar/tab-right}
}
```

## Support

For questions, issues, or feature requests, please use the [GitHub issue tracker](https://github.com/DanielAvdar/tab-right/issues).
