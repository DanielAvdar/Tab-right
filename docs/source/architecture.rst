.. _architecture:

Architecture
============

This page contains architecture diagrams that illustrate the structure and relationships between Tab-right's main components.

Core Components
---------------

Tab-right consists of several core modules and classes that work together to provide data analysis functionality:

.. mermaid::

   classDiagram
      class BaseSegmentationCalc {
          +df: pd.DataFrame
          +label_col: str
          +prediction_col: str
          +__call__(metric: Callable) -> pd.DataFrame
      }

      class DoubleSegmentation {
          +df: pd.DataFrame
          +label_col: str
          +prediction_col: str
          +_group_2_features(feature1, feature2, bins_1, bins_2) -> BaseSegmentationCalc
          +__call__(feature1_col, feature2_col, score_metric, bins_1, bins_2) -> pd.DataFrame
      }

      class DriftCalcP {
          +df1: pd.DataFrame
          +df2: pd.DataFrame
          +kind: Optional[Dict[str, str]]
          +__call__(columns, bins) -> pd.DataFrame
          +get_prob_density(columns, bins) -> pd.DataFrame
          +_categorical_drift_calc(s1, s2) -> float
          +_continuous_drift_calc(s1, s2, bins) -> float
      }

      class DriftPlotP {
          +drift_calc: DriftCalcP
          +plot_multiple(columns, bins, figsize, sort_by, ascending, top_n, threshold) -> Figure
          +plot_single(column, bins, figsize, show_metrics) -> Figure
          +get_distribution_plots(columns, bins) -> Dict[str, Figure]
      }

      class DoubleSegmPlottingP {
          +df: pd.DataFrame
          +metric_name: str
          +lower_is_better: bool
          +get_heatmap_df() -> pd.DataFrame
          +plot_heatmap() -> Figure
      }

      DoubleSegmentation --|> BaseSegmentationCalc : uses
      DriftPlotP --|> DriftCalcP : uses
      DoubleSegmPlottingP ..> DoubleSegmentation : uses results from

Module Structure
----------------

The following diagram shows the high-level module organization of Tab-right:

.. mermaid::

   graph TD
      A[tab_right] --> B[base_architecture]
      A --> C[segmentations]
      A --> D[drift]
      A --> E[plotting]
      A --> F[task_detection]

      B --> B1[seg_protocols.py]
      B --> B2[seg_plotting_protocols.py]
      B --> B3[drift_protocols.py]
      B --> B4[drift_plot_protocols.py]

      C --> C1[calc_seg.py]
      C --> C2[double_seg.py]

      D --> D1[drift_calculator.py]
      D --> D2[univariate.py]
      D --> D3[psi.py]
      D --> D4[cramer_v.py]

      E --> E1[plot_segmentations.py]
      E --> E2[drift_plotter.py]
      E --> E3[plot_drift.py]
      E --> E4[plot_feature_drift.py]

Protocol Relationships
----------------------

The following diagram illustrates the relationships between the main protocol interfaces:

.. mermaid::

   flowchart LR
      A[BaseSegmentationCalc] --> B[SegmentationCalc]
      A --> C[DoubleSegmentation]
      C --> D[DoubleSegmentationImp]

      E[DoubleSegmPlottingP] --> F[DoubleSegmPlotting]

      H[DriftCalcP] --> I[DriftCalculator]

      J[DriftPlotP] --> K[DriftPlotter]

      B -.-> F
      D -.-> F
      I -.-> K

Data Flow
---------

This diagram shows the typical data flow when using Tab-right:

.. mermaid::

   sequenceDiagram
      participant User
      participant Segmentation
      participant Metrics
      participant Plotting

      User->>Segmentation: Create segmentation with df, labels, predictions
      Segmentation->>Segmentation: Group data by features
      Segmentation->>Metrics: Calculate metrics per segment
      Metrics->>Segmentation: Return segment metrics
      Segmentation->>User: Return segmentation results
      User->>Plotting: Create visualization with results
      Plotting->>User: Return charts/figures

How to Update These Diagrams
----------------------------

These architecture diagrams can be updated by modifying the Mermaid syntax directly in this file. To update:

1. Edit this file (`architecture.rst`)
2. Update the Mermaid diagram code between the `.. mermaid::` directive blocks
3. Run `make html` to preview changes
4. Run `make doctest` to verify documentation integrity

For more information on Mermaid syntax, visit the `Mermaid documentation <https://mermaid.js.org/>`_.
