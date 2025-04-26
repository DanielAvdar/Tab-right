import plotly.graph_objects as go
import pandas as pd
from typing import Any, Optional

def plot_segmentation_heatmap(
    z: pd.DataFrame,
    x_labels: Optional[list] = None,
    y_labels: Optional[list] = None,
    title: str = "Segmentation Heatmap",
    zmin: float = None,
    zmax: float = None,
    annotation_fmt: str = ".2f",
    colorbar_title: str = "Score",
    colorscale: str = "RdYlGn_r"
) -> go.Figure:
    """
    Plot a segmentation heatmap with value annotations using Plotly.

    Parameters
    ----------
    z : pd.DataFrame
        2D DataFrame of scores (rows: y_labels, columns: x_labels).
    x_labels : list, optional
        Labels for the x-axis (columns). If None, use z.columns.
    y_labels : list, optional
        Labels for the y-axis (rows). If None, use z.index.
    title : str, default "Segmentation Heatmap"
        Title for the plot.
    zmin, zmax : float, optional
        Min/max for color scale. If None, use data min/max.
    annotation_fmt : str, default ".2f"
        Format string for value annotations.
    colorbar_title : str, default "Score"
        Title for the colorbar.
    colorscale : str, default "RdYlGn_r"
        Plotly colorscale name.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure with the heatmap.
    """
    if x_labels is None:
        x_labels = list(z.columns)
    if y_labels is None:
        y_labels = list(z.index)
    z_values = z.values
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=colorbar_title),
            hovertemplate="%{y} | %{x}: %{z:.2f}<extra></extra>",
        )
    )
    # Add value annotations
    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            val = z_values[i, j]
            fig.add_annotation(
                x=x,
                y=y,
                text=format(val, annotation_fmt),
                showarrow=False,
                font=dict(color="black", size=12, family="monospace"),
                xanchor="center",
                yanchor="middle",
            )
    fig.update_layout(
        title=title,
        xaxis_title="Feature 2",
        yaxis_title="Feature 1",
        xaxis=dict(constrain='domain'),
        yaxis=dict(autorange="reversed", constrain='domain'),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig
