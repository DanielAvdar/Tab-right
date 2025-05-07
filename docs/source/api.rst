.. _api:

API Reference
=============

Segmentation
------------

.. autoclass:: tab_right.segmentations.calc_seg.SegmentationCalc
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: tab_right.segmentations.double_seg.DoubleSegmentationImp
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Drift Detection
---------------

.. autoclass:: tab_right.drift.drift_calculator.DriftCalculator
   :members:
   :undoc-members:
   :show-inheritance:

Drift Metrics
~~~~~~~~~~~~~

.. autofunction:: tab_right.drift.univariate.detect_univariate_drift_df

.. autofunction:: tab_right.drift.univariate.detect_univariate_drift

.. autofunction:: tab_right.drift.psi.psi

.. autofunction:: tab_right.drift.cramer_v.cramer_v

Plotting Functions
------------------

Segmentation Plotting
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tab_right.plotting.plot_segmentations.plot_single_segmentation_impl

.. autofunction:: tab_right.plotting.plot_segmentations.plot_single_segmentation

.. autofunction:: tab_right.plotting.plot_segmentations.plot_single_segmentation_mp

.. autoclass:: tab_right.plotting.plot_segmentations.DoubleSegmPlotting
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tab_right.plotting.plot_segmentations.DoubleSegmPlottingMp
   :members:
   :undoc-members:
   :show-inheritance:

Drift Plotting
~~~~~~~~~~~~~~

.. autoclass:: tab_right.plotting.drift_plotter.DriftPlotter
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: tab_right.plotting.plot_drift.plot_drift

.. autofunction:: tab_right.plotting.plot_drift.plot_drift_mp

.. autofunction:: tab_right.plotting.plot_feature_drift.plot_feature_drift

.. autofunction:: tab_right.plotting.plot_feature_drift.plot_feature_drift_mp

Base Architecture & Protocols
-----------------------------

.. automodule:: tab_right.base_architecture.seg_protocols
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tab_right.base_architecture.seg_plotting_protocols
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tab_right.base_architecture.drift_protocols
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tab_right.base_architecture.drift_plot_protocols
   :members:
   :undoc-members:
   :show-inheritance:

Task Detection
--------------

.. automodule:: tab_right.task_detection
   :members:
   :undoc-members:
   :show-inheritance:
