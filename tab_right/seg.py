"""Tab-right package init."""

from dataclasses import dataclass
import pandas as pd


@dataclass
class SegmentationStats:
    df: pd.DataFrame
    label_col: str
    pred_col: str

    def run(self) -> pd.DataFrame:
        """Returns a DataFrame with segmentation statistics:
        - count
        - mean label
        - mean prediction
        - std of prediction
        - accuracy (if label is binary).
        """
        result = (
            self.df.groupby(self.pred_col)
            .agg(
                count=(self.label_col, "count"),
                mean_label=(self.label_col, "mean"),
                mean_pred=(self.pred_col, "mean"),
                std_pred=(self.pred_col, "std"),
            )
            .reset_index()
        )
        # If label is binary, add accuracy
        if set(self.df[self.label_col].dropna().unique()).issubset({0, 1}):
            result["accuracy"] = (
                self.df.groupby(self.pred_col)
                .apply(lambda g: (g[self.label_col] == g[self.pred_col].round()).mean())
                .values
            )
        return result

    def plot(self, feature: str, bins: int = 10, category_limit: int = 20, backend: str = "plotly", **kwargs):
        """Plots the distribution of the predictions and the label for a specific feature.
        Args:
            feature: The feature to plot.
            bins: The number of bins to use for the histogram.
            category_limit: The maximum number of categories to plot (if applicable).
            backend: The plotting backend to use (default is 'plotly').
            **kwargs: Additional arguments passed to the plotting function.
        """
        import matplotlib.pyplot as plt

        if self.df[feature].nunique() > category_limit:
            self.df[feature] = pd.qcut(self.df[feature], q=bins, duplicates="drop")
        else:
            self.df[feature] = pd.cut(self.df[feature], bins=bins)

        if backend == "plotly":
            import plotly.express as px

            fig = px.histogram(
                self.df,
                x=feature,
                color=self.label_col,
                barmode="overlay",
                histnorm="probability",
                **kwargs,
            )
            fig.update_traces(opacity=0.75)
            fig.show()
        elif backend == "matplotlib":
            self.df.groupby(feature).mean()[self.pred_col].plot(kind="bar")
            plt.show()
        else:
            raise ValueError(f"Unknown backend: {backend}")
