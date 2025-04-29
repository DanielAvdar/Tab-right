"""Decision tree based segmentation analysis for model errors."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


@dataclass
class DecisionTreeSegmentation:
    """DecisionTreeSegmentation provides error analysis using decision trees.

    This class uses decision trees to segment the feature space into regions
    with similar error characteristics based on two features.

    Parameters
    ----------
    df : Optional[pd.DataFrame], default=None
        Input DataFrame containing features and errors
    error_col : Optional[str], default=None
        Column name for error values
    feature_1_col : Optional[str], default=None
        Column name for first feature
    feature_2_col : Optional[str], default=None
        Column name for second feature
    tree_model : Optional[DecisionTreeRegressor], default=None
        Fitted decision tree model for segmentation
    metric : Optional[Callable], default=None
        Metric function to compute on segments
    max_depth : int, default=5
        Maximum depth of the decision tree. Controls the granularity of the segmentation.
    min_samples_leaf : int, default=20
        Minimum number of samples required in a leaf node. Prevents overfitting to outliers.

    """

    df: Optional[pd.DataFrame] = None
    error_col: Optional[str] = None
    feature_1_col: Optional[str] = None
    feature_2_col: Optional[str] = None
    tree_model: Optional[DecisionTreeRegressor] = None
    metric: Optional[Callable] = None
    max_depth: int = 5
    min_samples_leaf: int = 20

    def fit(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None,
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
            Names of the two features to use for segmentation analysis.
            If None and X is a DataFrame, the first two columns will be used.

        Returns
        -------
        self : DecisionTreeSegmentation
            The fitted object

        Notes
        -----
        This method trains a decision tree to predict the absolute error between
        y_true and y_pred based on the two selected features.

        """
        # Set feature names
        if isinstance(x, pd.DataFrame):
            if feature_names is None:
                feature_names = x.columns[:2].tolist()
            x_subset = x[feature_names].values
            self.feature_1_col = feature_names[0]
            self.feature_2_col = feature_names[1]
        else:
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(2)]
            else:
                feature_names = feature_names[:2]
            x_subset = x[:, :2]
            self.feature_1_col = feature_names[0]
            self.feature_2_col = feature_names[1]

        # Calculate errors (absolute difference between true and predicted values)
        errors = np.abs(y_true - y_pred)
        self.error_col = "abs_error"

        # Create DataFrame with features and errors
        self.df = pd.DataFrame({
            self.feature_1_col: x_subset[:, 0],
            self.feature_2_col: x_subset[:, 1],
            self.error_col: errors,
        })

        # Train a decision tree to predict errors
        self.tree_model = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.tree_model.fit(x_subset, errors)

        return self

    def get_segment_df(self, n_segments: int = 10) -> pd.DataFrame:
        """Get DataFrame with segment assignments and statistics.

        Parameters
        ----------
        n_segments : int, default=10
            Number of top segments to return

        Returns
        -------
        DataFrame
            DataFrame with original data plus segment assignments

        Raises
        ------
        ValueError
            If the model has not been fitted yet

        """
        if self.tree_model is None or self.df is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create a copy of the DataFrame
        result_df = self.df.copy()

        # Get segment assignments (leaf node IDs)
        x_subset = result_df[[self.feature_1_col, self.feature_2_col]].values
        leaf_ids = self.tree_model.apply(x_subset)
        result_df["segment_id"] = leaf_ids

        # Add segment statistics
        segment_stats = self.get_segment_stats(n_segments=n_segments)
        result_df = pd.merge(
            result_df,
            segment_stats[["segment_id", "mean_error", "median_error", "max_error"]],
            on="segment_id",
            how="left",
        )

        return result_df

    def get_segment_stats(self, n_segments: int = 10) -> pd.DataFrame:
        """Get statistics for the top error segments.

        Parameters
        ----------
        n_segments : int, default=10
            Number of top segments to return

        Returns
        -------
        DataFrame
            DataFrame with segment statistics including:
            - segment_id: Unique ID for the segment (leaf node ID)
            - size: Number of samples in the segment
            - mean_error: Average error in the segment
            - median_error: Median error in the segment
            - max_error: Maximum error in the segment
            - size_percent: Percentage of total samples in the segment

        Raises
        ------
        ValueError
            If the model has not been fitted yet

        """
        if self.tree_model is None or self.df is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get feature data
        x_subset = self.df[[self.feature_1_col, self.feature_2_col]].values

        # Get leaf node assignments for each data point
        leaf_ids = self.tree_model.apply(x_subset)

        # Calculate errors
        errors = self.df[self.error_col].values

        # Group by leaf node and calculate statistics
        results = []
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            segment_size = np.sum(mask)
            segment_errors = errors[mask]

            results.append({
                "segment_id": int(leaf_id),
                "size": int(segment_size),
                "mean_error": float(np.mean(segment_errors)),
                "median_error": float(np.median(segment_errors)),
                "max_error": float(np.max(segment_errors)),
                "size_percent": float(100 * segment_size / len(errors)),
            })

        # Convert to DataFrame and sort by mean error
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("mean_error", ascending=False).reset_index(drop=True)

        return results_df.head(n_segments)

    def get_decision_rules(self, n_segments: int = 10) -> Dict[int, List[Dict[str, Any]]]:
        """Extract human-readable decision rules for the top error segments.

        Parameters
        ----------
        n_segments : int, default=10
            Number of top segments to return rules for

        Returns
        -------
        Dict[int, List[Dict[str, Any]]]
            Dictionary mapping segment IDs to lists of rule dictionaries.
            Each rule dictionary contains:
            - feature: Feature name
            - operator: Comparison operator ('≤' or '>')
            - threshold: Split threshold value

        Raises
        ------
        ValueError
            If the model has not been fitted yet

        """
        if self.tree_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        tree = self.tree_model.tree_
        feature_names = [self.feature_1_col, self.feature_2_col]

        # Get the leaf nodes with highest error values
        leaf_nodes = []
        for i in range(tree.node_count):
            if tree.children_left[i] == tree.children_right[i]:  # It's a leaf
                leaf_nodes.append((i, tree.value[i][0][0]))  # (node_id, error_value)

        # Sort leaf nodes by error value (descending) and take top n
        leaf_nodes.sort(key=lambda x: x[1], reverse=True)
        top_leaf_nodes = leaf_nodes[:n_segments]

        # For each top leaf node, extract the path from root
        rules_by_segment = {}
        for node_id, _ in top_leaf_nodes:
            # Get the path from root to this leaf
            path = self._get_path_to_node(node_id)
            rules = []

            # Convert path to human-readable rules
            for i in range(len(path) - 1):
                parent, child = path[i], path[i + 1]

                # Get the feature and threshold for this split
                feature_idx = tree.feature[parent]
                threshold = tree.threshold[parent]

                # Determine if we went left (≤) or right (>)
                if child == tree.children_left[parent]:
                    operator = "≤"
                else:
                    operator = ">"

                # Handle case where feature_idx is out of range
                if feature_idx < len(feature_names):
                    feature_name = feature_names[feature_idx]
                else:
                    feature_name = f"Feature {feature_idx}"

                rules.append({"feature": feature_name, "operator": operator, "threshold": float(threshold)})

            rules_by_segment[int(node_id)] = rules

        return rules_by_segment

    def _get_path_to_node(self, node_id: int) -> List[int]:
        """Get the path from root to the given node.

        Parameters
        ----------
        node_id : int
            ID of the target node

        Returns
        -------
        List[int]
            List of node IDs in the path from root to target

        """
        path = [node_id]
        tree = self.tree_model.tree_

        # Traverse up the tree until we reach the root (node 0)
        current = node_id
        while current != 0:
            # Find the parent of the current node
            found = False
            for i in range(tree.node_count):
                if tree.children_left[i] == current or tree.children_right[i] == current:
                    path.append(i)
                    current = i
                    found = True
                    break
            if not found:
                break  # Should not happen with a valid tree

        # Reverse to get path from root to node
        return list(reversed(path))

    def get_feature_ranges(self) -> List[Tuple[float, float]]:
        """Get the min and max values for each feature with padding.

        Returns
        -------
        ranges : list of tuples
            List of (min, max) tuples for each feature

        Raises
        ------
        ValueError
            If the model has not been fitted yet

        """
        if self.tree_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Since sklearn trees don't store min/max feature values directly,
        # we need to compute the ranges from the decision thresholds
        tree = self.tree_model.tree_

        # Initialize with extreme values
        x_min, x_max = float("inf"), float("-inf")
        y_min, y_max = float("inf"), float("-inf")

        # Examine all nodes with valid thresholds
        for i in range(tree.node_count):
            # Skip leaf nodes (they don't have split thresholds)
            if tree.children_left[i] == tree.children_right[i]:
                continue

            # Get feature index for this node
            feature_idx = tree.feature[i]
            threshold = tree.threshold[i]

            # Update min/max for the appropriate feature
            if feature_idx == 0:  # First feature
                x_min = min(x_min, threshold)
                x_max = max(x_max, threshold)
            elif feature_idx == 1:  # Second feature
                y_min = min(y_min, threshold)
                y_max = max(y_max, threshold)

        # If we didn't find any splits on a feature, use reasonable defaults
        if x_min == float("inf") or x_max == float("-inf"):
            x_min, x_max = -1.0, 1.0
        if y_min == float("inf") or y_max == float("-inf"):
            y_min, y_max = -1.0, 1.0

        # Add padding (20% on each side to ensure we see all regions)
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_min -= 0.2 * x_range
        x_max += 0.2 * x_range
        y_min -= 0.2 * y_range
        y_max += 0.2 * y_range

        return [(x_min, x_max), (y_min, y_max)]
