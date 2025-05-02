"""Decision tree based segmentation analysis for model errors."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import BaseDecisionTree, DecisionTreeRegressor

from tab_right.base_architecture.seg_protocols import FindSegmentation


@dataclass
class DecisionTreeSegmentation(FindSegmentation):
    """Find feature segmentation using decision trees, implementing FindSegmentation protocol.

    Parameters
    ----------
    df : Optional[pd.DataFrame], default=None
        DataFrame containing the data to be segmented.
    feature1_col : Optional[str], default=None
        Column name for first feature
    feature2_col : Optional[str], default=None
        Column name for second feature
    error_col : Optional[str], default=None
        Column name for error values
    label_col : Optional[str], default=None
        Column name for the true target values.
    prediction_col : Optional[Union[str, List[str]]], default=None
        Column names for the predicted values. Can be a single column or a list of columns.
    max_depth : int, default=5
        Maximum depth of the decision tree
    min_samples_leaf : int, default=20
        Minimum number of samples required in a leaf node
    tree_model : Optional[BaseDecisionTree], default=None
        Fitted decision tree model for segmentation

    """

    # Parameters required by the FindSegmentation protocol with defaults
    # for backward compatibility
    df: Optional[pd.DataFrame] = None
    label_col: Optional[str] = None
    prediction_col: Optional[Union[str, List[str]]] = None

    # Additional parameters specific to this implementation
    feature1_col: Optional[str] = None
    feature2_col: Optional[str] = None
    error_col: Optional[str] = None
    max_depth: int = 5
    min_samples_leaf: int = 20
    tree_model: Optional[BaseDecisionTree] = None

    @property
    def feature_1_col(self) -> Optional[str]:
        """Backward compatibility accessor for feature1_col."""
        return self.feature1_col

    @feature_1_col.setter
    def feature_1_col(self, value: str) -> None:
        """Backward compatibility setter for feature1_col."""
        self.feature1_col = value

    @property
    def feature_2_col(self) -> Optional[str]:
        """Backward compatibility accessor for feature2_col."""
        return self.feature2_col

    @feature_2_col.setter
    def feature_2_col(self, value: str) -> None:
        """Backward compatibility setter for feature2_col."""
        self.feature2_col = value

    def fit(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "DecisionTreeSegmentation":
        """Fit the decision tree to predict errors based on two features.

        Parameters
        ----------
        x : DataFrame or array
            The features (only the two selected features will be used)
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values from another model
        feature_names : list of str, optional
            Names of the two features to use for segmentation analysis

        Returns
        -------
        self : DecisionTreeSegmentation
            The fitted object

        """
        # Set feature names
        if isinstance(x, pd.DataFrame):
            if feature_names is None:
                feature_names = x.columns[:2].tolist()
            x_subset = x[feature_names].values
        else:
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(2)]
            x_subset = x[:, :2]

        self.feature1_col = feature_names[0]
        self.feature2_col = feature_names[1]

        # Calculate errors
        errors = np.abs(y_true - y_pred)
        self.error_col = "abs_error"

        # Create DataFrame
        self.df = pd.DataFrame({
            self.feature1_col: x_subset[:, 0],
            self.feature2_col: x_subset[:, 1],
            self.error_col: errors,
        })

        # Add target and prediction columns if they aren't set
        if self.label_col is None:
            self.label_col = "y_true"
            self.df[self.label_col] = y_true

        if self.prediction_col is None:
            self.prediction_col = "y_pred"
            self.df[self.prediction_col] = y_pred

        # Train decision tree
        self.tree_model = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.train_tree_model(self.tree_model)

        return self

    @classmethod
    def _calc_error(
        cls,
        metric: Callable[[pd.Series, pd.DataFrame], pd.Series],
        y_true: pd.Series,
        y_pred: pd.DataFrame,
    ) -> pd.Series:
        """Calculate the error metric for each group in the DataFrame.

        Parameters
        ----------
        metric : Callable[[pd.Series, pd.DataFrame], pd.Series]
            A function that takes a pandas Series (true values) and a DataFrame (predicted values)
            and returns a Series representing the error metric for each row in the DataFrame.
        y_true : pd.Series
            The true target values.
        y_pred : pd.DataFrame
            The predicted values for each group, can be probabilities (multiple columns)
             or classes or continuous values.

        Returns
        -------
        pd.Series
            Series of error values for each row in the DataFrame.

        """
        return metric(y_true, y_pred)

    @classmethod
    def _fit_model(
        cls,
        model: BaseDecisionTree,
        feature: pd.Series,
        error: pd.Series,
    ) -> BaseDecisionTree:
        """Fit the decision tree model to the feature and error data.

        Parameters
        ----------
        model : BaseDecisionTree
            The decision tree model to fit.
        feature : pd.Series
            The feature data to use for fitting the model.
        error : pd.Series
            The error calculated for each row in the DataFrame, which is used as the target variable.

        Returns
        -------
        BaseDecisionTree
            The fitted decision tree model.

        """
        # Reshape feature to be 2D if it's 1D
        if len(feature.shape) == 1:
            feature_array = feature.values.reshape(-1, 1)
        else:
            feature_array = feature.values

        model.fit(feature_array, error.values)
        return model

    @classmethod
    def _extract_leaves(
        cls,
        model: BaseDecisionTree,
    ) -> pd.DataFrame:
        """Extract the leaves of the fitted decision tree model.

        Parameters
        ----------
        model : BaseDecisionTree
            The fitted decision tree model to extract leaves from.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the leaves of the decision tree.
            columns:
            - `segment_id`: The ID of the segment, for grouping.
            - `segment_name`: (str) the range or category of the first feature.

        """
        tree = model.tree_
        leaf_ids = []

        # Get leaf nodes
        for i in range(tree.node_count):
            if tree.children_left[i] == tree.children_right[i]:  # Leaf node
                leaf_ids.append(i)

        # Create segment names based on decision paths
        segments = []
        for leaf_id in leaf_ids:
            # This is a simplification - in a real implementation, you would
            # trace the path from root to leaf to generate a meaningful name
            segment_name = f"Segment {leaf_id}"
            segments.append({"segment_id": leaf_id, "segment_name": segment_name})

        return pd.DataFrame(segments)

    def train_tree_model(self, model: BaseDecisionTree) -> BaseDecisionTree:
        """Train the decision tree model.

        Parameters
        ----------
        model : BaseDecisionTree
            The decision tree model to train

        Returns
        -------
        BaseDecisionTree
            The trained model

        Raises
        ------
        ValueError
            If DataFrame or required column names are not set

        """
        if self.df is None or self.feature1_col is None or self.feature2_col is None or self.error_col is None:
            raise ValueError("DataFrame and column names must be set before training")

        x_subset = self.df[[self.feature1_col, self.feature2_col]].values
        errors = self.df[self.error_col].values

        model.fit(x_subset, errors)
        return model

    def __call__(
        self,
        model: Optional[BaseDecisionTree] = None,
        feature_col: Optional[str] = None,
        error_metric: Optional[Callable[[pd.Series, pd.DataFrame], pd.Series]] = None,
    ) -> DataFrameGroupBy:
        """Apply the model to segment the DataFrame.

        This implementation maintains backward compatibility with the existing code
        while also allowing the protocol-compatible usage with feature_col and error_metric.

        Parameters
        ----------
        model : BaseDecisionTree, optional
            The decision tree model to use for segmentation
        feature_col : str, optional
            The name of the feature to segment by, required for protocol-compatible usage
        error_metric : Callable[[pd.Series, pd.DataFrame], pd.Series], optional
            A function that calculates error metrics, required for protocol-compatible usage

        Returns
        -------
        DataFrameGroupBy
            DataFrame grouped by segment ID

        Raises
        ------
        ValueError
            If DataFrame is not set or model is not provided/fitted

        """
        if self.df is None:
            raise ValueError("DataFrame must be set before calling")

        if model is not None:
            self.tree_model = model

        if self.tree_model is None:
            raise ValueError("Model not fitted. Call fit() or provide a model.")

        # Protocol-compatible usage (overrides the default behavior)
        if feature_col is not None and error_metric is not None:
            # Use the provided feature column
            feature_data = self.df[feature_col]

            # Calculate error using the provided metric
            y_true = self.df[self.label_col] if self.label_col is not None else self.df.iloc[:, 0]

            # Handle different formats of prediction_col
            if isinstance(self.prediction_col, list):
                y_pred = self.df[self.prediction_col]
            elif self.prediction_col is not None:
                y_pred = self.df[[self.prediction_col]]
            else:
                y_pred = self.df.iloc[:, 1:2]

            # Calculate error for each row
            errors = self._calc_error(error_metric, y_true, y_pred)

            # Fit the model to the feature and errors
            fitted_model = self._fit_model(self.tree_model, feature_data, errors)

            # Extract leaf information
            self._extract_leaves(fitted_model)

            # Assign segments to the data
            if len(feature_data.shape) == 1:
                feature_array = feature_data.values.reshape(-1, 1)
            else:
                feature_array = feature_data.values

            leaf_ids = fitted_model.apply(feature_array)

            # Create result DataFrame with segment assignments
            result_df = self.df.copy()
            result_df["segment_id"] = leaf_ids

            # Return grouped DataFrame
            return result_df.groupby("segment_id")

        # Original functionality (backward compatibility)
        result_df = self.df.copy()
        x_subset = result_df[[self.feature1_col, self.feature2_col]]
        leaf_ids = self.tree_model.apply(x_subset)
        result_df["segment_id"] = leaf_ids

        # Return grouped DataFrame
        return result_df.groupby("segment_id")

    def get_feature_ranges(self) -> List[Tuple[float, float]]:
        """Get the range of values for each feature in the tree.

        Returns
        -------
        List[Tuple[float, float]]
            List of (min, max) tuples for each feature

        Raises
        ------
        ValueError
            If tree model is not fitted

        """
        if self.tree_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        tree = self.tree_model.tree_
        x_min = float("inf")
        x_max = float("-inf")
        y_min = float("inf")
        y_max = float("-inf")

        # Iterate through all nodes to find feature ranges
        for i in range(tree.node_count):
            if tree.children_left[i] == tree.children_right[i]:
                continue

            feature_idx = tree.feature[i]
            threshold = tree.threshold[i]

            if feature_idx == 0:
                x_min = min(x_min, threshold)
                x_max = max(x_max, threshold)
            elif feature_idx == 1:
                y_min = min(y_min, threshold)
                y_max = max(y_max, threshold)

        # Use reasonable defaults if no splits found
        if x_min == float("inf") or x_max == float("-inf"):
            x_min, x_max = -1.0, 1.0
        if y_min == float("inf") or y_max == float("-inf"):
            y_min, y_max = -1.0, 1.0

        # Add padding (20% on each side)
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.2 * x_range
        x_max += 0.2 * x_range
        y_min -= 0.2 * y_range
        y_max += 0.2 * y_range

        return [(x_min, x_max), (y_min, y_max)]

    def get_segment_df(self, n_segments: int = 10) -> pd.DataFrame:
        """Get DataFrame with segment information.

        Parameters
        ----------
        n_segments : int, default=10
            Number of segments to include

        Returns
        -------
        pd.DataFrame
            DataFrame containing segment information and feature statistics

        Raises
        ------
        ValueError
            If tree model is not fitted

        """
        if self.tree_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        gdf = self.__call__()

        # Create aggregation dict including feature columns
        agg_dict = {
            self.error_col: ["mean", "median", "max", "count"],
            self.feature1_col: "mean",
            self.feature2_col: "mean",
        }

        # Get aggregated stats
        result_df = gdf.agg(agg_dict)

        # Handle multi-level columns correctly
        result_df.columns = result_df.columns.map(lambda x: x[0] if x[1] == "" else f"{x[0]}_{x[1]}")
        result_df = result_df.reset_index()

        # Rename columns to match expected format
        rename_map = {
            f"{self.error_col}_mean": "mean_error",
            f"{self.error_col}_median": "median_error",
            f"{self.error_col}_max": "max_error",
            f"{self.error_col}_count": "size",
            f"{self.feature1_col}_mean": self.feature1_col,
            f"{self.feature2_col}_mean": self.feature2_col,
        }
        result_df = result_df.rename(columns=rename_map)

        # Sort by mean error and take top n segments
        result_df = result_df.sort_values("mean_error", ascending=False).head(n_segments)

        # Add error column with mean error values for compatibility
        result_df[self.error_col] = result_df["mean_error"]

        return result_df

    def get_segment_stats(self, n_segments: int = 10) -> pd.DataFrame:
        """Get statistics for each segment.

        Parameters
        ----------
        n_segments : int, default=10
            Number of segments to include

        Returns
        -------
        pd.DataFrame
            DataFrame containing segment statistics

        Raises
        ------
        ValueError
            If tree model is not fitted

        """
        return self.get_segment_df(n_segments)

    def _get_path_to_node(self, node_id: int) -> List[int]:
        """Get path from root to specified node.

        Parameters
        ----------
        node_id : int
            ID of the target node

        Returns
        -------
        List[int]
            List of node IDs from root to target node

        Raises
        ------
        ValueError
            If tree model is not fitted

        """
        if self.tree_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        path = []
        current_id = node_id

        # Walk up the tree until we reach the root
        while current_id != 0:
            path.append(current_id)
            parent_id = -1

            # Find the parent node
            for i in range(self.tree_model.tree_.node_count):
                if (
                    self.tree_model.tree_.children_left[i] == current_id
                    or self.tree_model.tree_.children_right[i] == current_id
                ):
                    parent_id = i
                    break

            if parent_id == -1:
                break

            current_id = parent_id

        path.append(0)
        return path[::-1]

    def get_decision_rules(self, n_segments: int = 10) -> Dict[int, List[Dict[str, Any]]]:
        """Get decision rules for each segment.

        Parameters
        ----------
        n_segments : int, default=10
            Number of segments to include

        Returns
        -------
        Dict[int, List[Dict[str, Any]]]
            Dictionary mapping segment IDs to their decision rules

        Raises
        ------
        ValueError
            If tree model is not fitted

        """
        if self.tree_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        tree = self.tree_model.tree_
        feature_names = [self.feature1_col, self.feature2_col]
        rules_by_segment = {}

        # Get leaf nodes with highest error
        leaf_nodes = []
        for i in range(tree.node_count):
            if tree.children_left[i] == tree.children_right[i]:
                leaf_nodes.append((i, tree.value[i][0][0]))

        # Sort by error value and take top n_segments
        leaf_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = leaf_nodes[:n_segments]

        # Get rules for each top segment
        for node_id, _ in top_nodes:
            path = self._get_path_to_node(node_id)
            rules = []

            for i in range(len(path) - 1):
                parent, child = path[i], path[i + 1]
                feature_idx = tree.feature[parent]
                threshold = tree.threshold[parent]

                if child == tree.children_left[parent]:
                    operator = "≤"
                else:
                    operator = ">"

                feature_name = (
                    feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
                )

                rules.append({"feature": feature_name, "operator": operator, "threshold": float(threshold)})

            rules_by_segment[int(node_id)] = rules

        return rules_by_segment
