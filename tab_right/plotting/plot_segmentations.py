"""Module for plotting segmentations of a DataFrame."""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from tab_right.segmentations.dt_segmentation import DecisionTreeSegmentation


def plot_segmentations(
    df: pd.DataFrame,
    good_color: str = "green",
    bad_color: str = "red",
    score_col: str = "score",
    segment_col: str = "segment",
    ascending: bool = False,
    fig: Optional[go.Figure] = None,
) -> go.Figure:
    """Plot the segmentations of a given DataFrame as a bar chart.

    Args:
        df (pd.DataFrame): The input data.
        good_color (str, optional): Color for good segments. Defaults to "green".
        bad_color (str, optional): Color for bad segments. Defaults to "red".
        score_col (str, optional): Column name for scores. Defaults to "score".
        segment_col (str, optional): Column name for segments. Defaults to "segment".
        ascending (bool, optional): Sort order. Defaults to False.
        fig (go.Figure, optional): Existing figure to update. Defaults to None.

    Returns:
        go.Figure: A plotly figure object.

    """
    if fig is None:
        fig = go.Figure()
    df = df.sort_values(by=score_col, ascending=ascending)
    for segment, group in df.groupby(segment_col):
        color = good_color if group[score_col].iloc[0] >= 0 else bad_color
        fig.add_trace(
            go.Bar(
                x=group.index,
                y=group[score_col],
                name=str(segment),
                marker_color=color,
            )
        )
    fig.update_layout(barmode="group")
    return fig


def plot_dt_segmentation(
    segmentation: "DecisionTreeSegmentation",
    resolution: int = 100,
    cmap: str = "RdYlGn_r",
    figsize: Tuple[int, int] = (800, 600),
    show: bool = True,
    metric_name: str = "MSE",
) -> go.Figure:
    """Create an interactive heatmap visualization of tree-based error segmentation.

    Parameters
    ----------
    segmentation : DecisionTreeSegmentation
        Fitted DecisionTreeSegmentation object with a trained tree model
    resolution : int, default=100
        Grid resolution for the heatmap. Higher values create smoother visualizations
        but increase computation time.
    cmap : str, default='RdYlGn_r'
        Colormap for the heatmap visualization. Default is green for good performance,
        red for bad performance.
    figsize : tuple, default=(800, 600)
        Figure size (width, height) in pixels
    show : bool, default=True
        Whether to display the plot immediately
    metric_name : str, default="MSE"
        Name of the error metric used (e.g., MSE, MAE, RMSE)

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object with the error heatmap

    Raises
    ------
    ValueError
        If the segmentation model has not been fitted

    """
    if segmentation.tree_model is None:
        raise ValueError("Model not fitted. Call fit() first.")

    # Get feature ranges
    feature_ranges = segmentation.get_feature_ranges()
    x_min, x_max = feature_ranges[0]
    y_min, y_max = feature_ranges[1]

    # Create a grid for visualization
    x_range = np.linspace(x_min, x_max, resolution)
    y_range = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_range, y_range)

    # Predict errors for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_predictions = segmentation.tree_model.predict(grid_points)

    # Reshape predictions to match the grid
    grid_errors = grid_predictions.reshape(xx.shape)

    # Create the heatmap visualization
    fig = go.Figure(
        data=go.Heatmap(
            z=grid_errors,
            x=x_range,
            y=y_range,
            colorscale=cmap,
            colorbar=dict(title=f"{metric_name} Error"),
            text=np.round(grid_errors, 3),  # Display score values rounded to 3 decimal places
            hovertemplate="%{x:.2f}, %{y:.2f}<br>Error: %{z:.3f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Error Segmentation by Decision Tree ({metric_name})",
        width=figsize[0],
        height=figsize[1],
        xaxis=dict(title=segmentation.feature1_col),  # Updated to use the new attribute name
        yaxis=dict(title=segmentation.feature2_col),  # Updated to use the new attribute name
    )

    if show:
        fig.show()

    return fig


def plot_dt_segmentation_with_stats(
    segmentation: "DecisionTreeSegmentation",
    resolution: int = 100,
    cmap: str = "RdYlGn_r",
    figsize: Tuple[int, int] = (1000, 500),
    n_top_segments: int = 5,
    show: bool = True,
    metric_name: str = "MSE",
) -> go.Figure:
    """Create a visualization with both error heatmap and top segment statistics.

    Parameters
    ----------
    segmentation : DecisionTreeSegmentation
        Fitted DecisionTreeSegmentation object with a trained tree model
    resolution : int, default=100
        Grid resolution for the heatmap
    cmap : str, default='RdYlGn_r'
        Colormap for the heatmap visualization. Green for good performance, red for bad.
    figsize : tuple, default=(1000, 500)
        Figure size (width, height) in pixels
    n_top_segments : int, default=5
        Number of top segments to show in the bar chart
    show : bool, default=True
        Whether to display the plot immediately
    metric_name : str, default="MSE"
        Name of the error metric used (e.g., MSE, MAE, RMSE)

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure with subplots containing heatmap and bar chart

    Raises
    ------
    ValueError
        If the segmentation model has not been fitted

    """
    if segmentation.tree_model is None:
        raise ValueError("Model not fitted. Call fit() first.")

    # Create figure with subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Error Segmentation Heatmap ({metric_name})", "Top Error Segments"),
        specs=[[{"type": "heatmap"}, {"type": "bar"}]],
        column_widths=[0.7, 0.3],
    )

    # Get feature ranges
    feature_ranges = segmentation.get_feature_ranges()
    x_min, x_max = feature_ranges[0]
    y_min, y_max = feature_ranges[1]

    # Create a grid for visualization
    x_range = np.linspace(x_min, x_max, resolution)
    y_range = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_range, y_range)

    # Predict errors for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_predictions = segmentation.tree_model.predict(grid_points)

    # Reshape predictions to match the grid
    grid_errors = grid_predictions.reshape(xx.shape)

    # Add heatmap to first subplot
    fig.add_trace(
        go.Heatmap(
            z=grid_errors,
            x=x_range,
            y=y_range,
            colorscale=cmap,
            colorbar=dict(title=f"{metric_name} Error", x=0.45),
            text=np.round(grid_errors, 3),  # Display score values rounded to 3 decimal places
            hovertemplate="%{x:.2f}, %{y:.2f}<br>Error: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Get segment statistics
    stats = segmentation.get_segment_stats(n_segments=n_top_segments)

    # Add bar chart to second subplot
    fig.add_trace(
        go.Bar(
            x=stats["segment_id"].astype(str),
            y=stats["mean_error"],
            text=stats["mean_error"].round(3).astype(str),  # Display the score value
            textposition="auto",
            marker=dict(
                color=stats["mean_error"],
                colorscale=cmap,
                colorbar=dict(title=f"{metric_name} Error", y=0.5, len=0.4, x=1.05),
            ),
            name=f"Mean {metric_name}",
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=f"Decision Tree Error Segmentation Analysis ({metric_name})", width=figsize[0], height=figsize[1]
    )

    # Update axes
    fig.update_xaxes(title_text=segmentation.feature1_col, row=1, col=1)  # Updated to use the new attribute name
    fig.update_yaxes(title_text=segmentation.feature2_col, row=1, col=1)  # Updated to use the new attribute name
    fig.update_xaxes(title_text="Segment ID", row=1, col=2)
    fig.update_yaxes(title_text=f"Mean {metric_name} Error", row=1, col=2)

    if show:
        fig.show()

    return fig
