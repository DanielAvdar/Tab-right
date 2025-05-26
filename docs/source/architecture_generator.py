"""
Helper script to generate architecture diagrams for tab-right documentation.

This script analyzes the tab_right package structure and generates Mermaid
diagrams that can be included in the architecture documentation.

Usage:
    python architecture_generator.py > architecture_diagrams.txt
"""

import os
import importlib
import inspect
import pkgutil
from typing import Dict, List, Set, Tuple, Optional


def find_modules(package_name: str) -> List[str]:
    """
    Find all modules within a package.
    
    Args:
        package_name: The name of the package to inspect
        
    Returns:
        A list of module names
    """
    package = importlib.import_module(package_name)
    modules = []
    
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        if is_pkg:
            modules.extend(find_modules(name))
        else:
            modules.append(name)
            
    return modules


def generate_module_structure() -> str:
    """
    Generate a Mermaid diagram of the module structure.
    
    Returns:
        A Mermaid diagram representation of the module structure
    """
    mermaid = [
        "graph TD",
        "    A[tab_right] --> B[base_architecture]",
        "    A --> C[segmentations]",
        "    A --> D[drift]",
        "    A --> E[plotting]",
        "    A --> F[task_detection]",
        "",
        "    B --> B1[seg_protocols.py]",
        "    B --> B2[seg_plotting_protocols.py]",
        "    B --> B3[drift_protocols.py]",
        "    B --> B4[drift_plot_protocols.py]",
        "",
        "    C --> C1[calc_seg.py]",
        "    C --> C2[double_seg.py]",
        "",
        "    D --> D1[drift_calculator.py]",
        "    D --> D2[univariate.py]",
        "    D --> D3[psi.py]",
        "    D --> D4[cramer_v.py]",
        "",
        "    E --> E1[plot_segmentations.py]",
        "    E --> E2[drift_plotter.py]",
        "    E --> E3[plot_drift.py]",
        "    E --> E4[plot_feature_drift.py]"
    ]
    
    return "\n".join(mermaid)


def generate_class_diagram() -> str:
    """
    Generate a Mermaid class diagram of the main protocols.
    
    Returns:
        A Mermaid class diagram representation
    """
    mermaid = [
        "classDiagram",
        "    class BaseSegmentationCalc {",
        "        +df: pd.DataFrame",
        "        +label_col: str",
        "        +prediction_col: str",
        "        +__call__(metric: Callable) -> pd.DataFrame",
        "    }",
        "    ",
        "    class DoubleSegmentation {",
        "        +df: pd.DataFrame",
        "        +label_col: str",
        "        +prediction_col: str",
        "        +_group_2_features(feature1, feature2, bins_1, bins_2) -> BaseSegmentationCalc",
        "        +__call__(feature1_col, feature2_col, score_metric, bins_1, bins_2) -> pd.DataFrame",
        "    }",
        "    ",
        "    class DriftCalcP {",
        "        +df1: pd.DataFrame",
        "        +df2: pd.DataFrame",
        "        +kind: Optional[Dict[str, str]]",
        "        +__call__(columns, bins) -> pd.DataFrame",
        "        +get_prob_density(columns, bins) -> pd.DataFrame",
        "        +_categorical_drift_calc(s1, s2) -> float",
        "        +_continuous_drift_calc(s1, s2, bins) -> float",
        "    }",
        "    ",
        "    class DriftPlotP {",
        "        +drift_calc: DriftCalcP",
        "        +plot_multiple(columns, bins, figsize, sort_by, ascending, top_n, threshold) -> Figure",
        "        +plot_single(column, bins, figsize, show_metrics) -> Figure",
        "        +get_distribution_plots(columns, bins) -> Dict[str, Figure]",
        "    }",
        "    ",
        "    class DoubleSegmPlottingP {",
        "        +df: pd.DataFrame",
        "        +metric_name: str",
        "        +lower_is_better: bool",
        "        +get_heatmap_df() -> pd.DataFrame",
        "        +plot_heatmap() -> Figure",
        "    }",
        "    ",
        "    DoubleSegmentation --|> BaseSegmentationCalc : uses",
        "    DriftPlotP --|> DriftCalcP : uses",
        "    DoubleSegmPlottingP ..> DoubleSegmentation : uses results from"
    ]
    
    return "\n".join(mermaid)


def generate_protocol_relationships() -> str:
    """
    Generate a Mermaid flowchart of protocol relationships.
    
    Returns:
        A Mermaid flowchart representation
    """
    mermaid = [
        "flowchart LR",
        "    A[BaseSegmentationCalc] --> B[SegmentationCalc]",
        "    A --> C[DoubleSegmentation]",
        "    C --> D[DoubleSegmentationImp]",
        "    ",
        "    E[DoubleSegmPlottingP] --> F[DoubleSegmPlotting]",
        "    E --> G[DoubleSegmPlottingMp]",
        "    ",
        "    H[DriftCalcP] --> I[DriftCalculator]",
        "    ",
        "    J[DriftPlotP] --> K[DriftPlotter]",
        "    ",
        "    B -.-> F",
        "    D -.-> F",
        "    I -.-> K"
    ]
    
    return "\n".join(mermaid)


def generate_sequence_diagram() -> str:
    """
    Generate a Mermaid sequence diagram showing data flow.
    
    Returns:
        A Mermaid sequence diagram representation
    """
    mermaid = [
        "sequenceDiagram",
        "    participant User",
        "    participant Segmentation",
        "    participant Metrics",
        "    participant Plotting",
        "    ",
        "    User->>Segmentation: Create segmentation with df, labels, predictions",
        "    Segmentation->>Segmentation: Group data by features",
        "    Segmentation->>Metrics: Calculate metrics per segment",
        "    Metrics->>Segmentation: Return segment metrics",
        "    Segmentation->>User: Return segmentation results",
        "    User->>Plotting: Create visualization with results",
        "    Plotting->>User: Return charts/figures"
    ]
    
    return "\n".join(mermaid)


if __name__ == "__main__":
    print("Module Structure Diagram:")
    print("-------------------------")
    print("```mermaid")
    print(generate_module_structure())
    print("```")
    print("\n")
    
    print("Class Diagram:")
    print("-------------")
    print("```mermaid")
    print(generate_class_diagram())
    print("```")
    print("\n")
    
    print("Protocol Relationships:")
    print("----------------------")
    print("```mermaid")
    print(generate_protocol_relationships())
    print("```")
    print("\n")
    
    print("Sequence Diagram:")
    print("----------------")
    print("```mermaid")
    print(generate_sequence_diagram())
    print("```")