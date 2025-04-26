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
![Last Commit](https://img.shields.io/github/last-commit/DanielAvdar/tab-right/main)

**tab-right** is a Python package designed to simplify the analysis of tabular data for inference models—both machine learning and non-ML. The core philosophy is that most analyses, such as segmentation strength, drift analysis, and feature predictive value, can be performed using model predictions alone, without direct access to the model itself. This approach enables powerful, model-agnostic diagnostics and interpretability, making the package easy to implement and use.

## Key features
- Analyze prediction strength across different data segments to uncover model biases and subgroup performance.
- Assess feature predictive power and value to inference, using techniques like feature importance, partial dependence, and more.
- Perform drift analysis and monitor changes in data or prediction distributions over time.
- Generate rich visualization reports for all analyses, supporting both interactive and static outputs.
- Focus on data and predictions, not model internals, for maximum flexibility and simplicity.
