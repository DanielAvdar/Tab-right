import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error

# Import required modules from tab_right
from tab_right.plotting.plot_segmentations import DoubleSegmPlotting
from tab_right.segmentations.double_seg import DoubleSegmentationImp

data = fetch_openml("adult", version=2, as_frame=True)
df = data.frame.copy()
df = df.sample(n=5000, random_state=42).reset_index(drop=True)  # Use a sample
df = df.dropna()  # Drop missing for simplicity

# Create dummy target and prediction columns
np.random.seed(42)
df["target"] = np.random.randint(0, 2, size=len(df))
df["prediction"] = np.random.rand(len(df))  # Dummy probability prediction

# Select relevant columns for analysis
df_analysis = df[
    ["age", "education-num", "hours-per-week", "target", "prediction"]
].copy()  # Add more features if needed
df_analysis.head()


double_segmentation_imp = DoubleSegmentationImp(
    df=df_analysis,
    label_col="target",
    prediction_col="prediction",  # Use the dummy prediction column
)


feature_pairs = [("age", "education-num"), ("age", "hours-per-week")]

# Analyze each feature pair and visualize
for feature1, feature2 in feature_pairs:
    # print(f"\nAnalyzing feature pair: {feature1} and {feature2}")

    # Calculate double segmentation scores using mean_squared_error
    # The __call__ method takes feature1_col, feature2_col, score_metric, bins_1, bins_2
    double_segments = double_segmentation_imp(
        feature1_col=feature1,
        feature2_col=feature2,
        score_metric=mean_squared_error,  # Use a metric compatible with dummy data
        bins_1=5,  # Define number of bins for numeric features
        bins_2=5,
    )

    # Create double segmentation plotter using the correct column name 'score'
    double_plotter = DoubleSegmPlotting(df=double_segments, metric_name="score")  # Use 'score' as metric_name

    # Plot the heatmap
    heatmap_fig = double_plotter.plotly_heatmap()
    heatmap_fig.update_layout(
        title=f"MSE Heatmap: {feature1} vs {feature2}", xaxis_title=feature1, yaxis_title=feature2
    )
    heatmap_fig.show()
