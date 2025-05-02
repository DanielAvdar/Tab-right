"""Decision tree based segmentation analysis for model errors."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import BaseDecisionTree, DecisionTreeRegressor


@dataclass
class DecisionTreeSegmentation:
    """DecisionTreeSegmentation provides error analysis using decision trees.

    This class uses decision trees to segment the feature space into regions
    with similar error characteristics based on two features.

    Parameters
    ----------
    df : Optional[pd.DataFrame], default=None
        Input DataFrame containing features and errors
    feature1_col : Optional[str], default=None
        Column name for first feature
    feature2_col : Optional[str], default=None
        Column name for second feature
    error_col : Optional[str], default=None
        Column name for error values
    tree_model : Optional[DecisionTreeRegressor], default=None
        Fitted decision tree model for segmentation
    max_depth : int, default=5
        Maximum depth of the decision tree
    min_samples_leaf : int, default=20
        Minimum number of samples required in a leaf node

    """

    df: Optional[pd.DataFrame] = None
    feature1_col: Optional[str] = None
    feature2_col: Optional[str] = None
    error_col: Optional[str] = None
    tree_model: Optional[DecisionTreeRegressor] = None
    max_depth: int = 5
    min_samples_leaf: int = 20

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

        # Train decision tree
        self.tree_model = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.train_tree_model(self.tree_model)

        return self

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

    def __call__(self, model: Optional[BaseDecisionTree] = None) -> DataFrameGroupBy:
        """Apply the model to segment the DataFrame.

        Parameters
        ----------
        model : BaseDecisionTree, optional
            The decision tree model to use for segmentation

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

        # Create a copy and add segment assignments
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
