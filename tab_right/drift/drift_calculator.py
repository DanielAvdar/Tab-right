"""Implementation of the DriftCalcP protocol."""

from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wasserstein_distance

from tab_right.base_architecture.drift_protocols import DriftCalcP


class DriftCalculator(DriftCalcP):
    """Implementation of DriftCalcP using Cramér's V and Wasserstein distance."""

    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame, kind: Union[str, Iterable[bool], Dict[str, str]] = "auto"):
        """Initialize the DriftCalculator with reference and current datasets.

        Args:
            df1: Reference DataFrame.
            df2: Current DataFrame for comparison.
            kind: Specification of feature types. Can be:
                - "auto": Automatically determine types
                - "categorical" or "continuous": Use this type for all features
                - Dict mapping column names to types
                - Iterable of booleans indicating if each column is continuous

        """
        self.df1 = df1
        self.df2 = df2
        self.kind = kind
        self._feature_types = self._determine_feature_types()

    def _determine_feature_types(self) -> Dict[str, str]:
        """Determine if features are categorical or continuous based on `kind`.

        Returns:
            Dictionary mapping column names to their types ("categorical" or "continuous").

        Raises:
            ValueError: If an invalid string value is provided for `kind` or if the
                length of the iterable doesn't match the number of columns.
            TypeError: If `kind` is not a string, dict, or iterable.

        """
        common_cols = list(set(self.df1.columns) & set(self.df2.columns))
        feature_types = {}

        if isinstance(self.kind, str):
            if self.kind == "auto":
                for col in common_cols:
                    if pd.api.types.is_numeric_dtype(self.df1[col]) and pd.api.types.is_numeric_dtype(self.df2[col]):
                        # Heuristic: Treat numeric with few unique values relative to size as categorical
                        if self.df1[col].nunique() < 20 or self.df2[col].nunique() < 20:
                            feature_types[col] = "categorical"
                        else:
                            feature_types[col] = "continuous"
                    else:
                        feature_types[col] = "categorical"
            elif self.kind in ["categorical", "continuous"]:
                feature_types = {col: self.kind for col in common_cols}
            else:
                raise ValueError("Invalid string value for `kind`.")
        elif isinstance(self.kind, dict):
            feature_types = {col: self.kind.get(col, "auto") for col in common_cols}
            # Resolve any remaining "auto" types
            auto_cols = [col for col, type_ in feature_types.items() if type_ == "auto"]
            auto_types = DriftCalculator(self.df1[auto_cols], self.df2[auto_cols], kind="auto")._feature_types
            feature_types.update(auto_types)
        elif isinstance(self.kind, Iterable) and not isinstance(self.kind, str):
            kind_list = list(self.kind)  # Convert to list to get length safely
            if len(kind_list) != len(common_cols):
                raise ValueError("Length of `kind` iterable must match number of common columns.")
            feature_types = {
                col: ("continuous" if is_cont else "categorical") for col, is_cont in zip(common_cols, kind_list)
            }
        else:
            raise TypeError("`kind` must be 'auto', 'categorical', 'continuous', a dict, or an iterable.")

        return feature_types

    def __call__(self, columns: Optional[Iterable[str]] = None, bins: int = 10, **kwargs: Any) -> pd.DataFrame:
        """Calculate drift metrics between the reference and current datasets.

        Args:
            columns: Specific columns to calculate drift for. If None, all common columns are used.
            bins: Number of bins to use for continuous features.
            **kwargs: Additional arguments passed to specific drift calculation methods.

        Returns:
            DataFrame with drift metrics for each feature, containing:
                - feature: Name of the feature
                - type: Type of metric used (cramer_v, wasserstein, or N/A)
                - score: Normalized drift score (for Wasserstein, this is the raw score)
                - raw_score: Unnormalized drift metric value

        Raises:
            ValueError: If an unknown column type is encountered.

        """
        if columns is None:
            columns = list(self._feature_types.keys())
        else:
            columns = [col for col in columns if col in self._feature_types]

        results = []
        for col in columns:
            s1 = self.df1[col].dropna()
            s2 = self.df2[col].dropna()
            col_type = self._feature_types[col]

            if s1.empty or s2.empty:
                score = np.nan
                metric_type = "N/A (Empty Data)"
                raw_score = np.nan
            elif col_type == "categorical":
                score = self._categorical_drift_calc(s1, s2)
                metric_type = "cramer_v"
                raw_score = score  # Already normalized
            elif col_type == "continuous":
                # Wasserstein distance is not naturally normalized to [0,1]
                # We return the raw score here. Normalization depends on context/range.
                raw_score = self._continuous_drift_calc(s1, s2, bins=bins)
                score = raw_score  # Placeholder: No standard normalization applied here
                metric_type = "wasserstein"
            else:
                raise ValueError(f"Unknown column type '{col_type}' for column '{col}'")

            results.append({
                "feature": col,
                "type": metric_type,
                "score": score,  # Note: Wasserstein score is not normalized here
                "raw_score": raw_score,
            })

        return pd.DataFrame(results)

    def get_prob_density(
        self,
        columns: Optional[Iterable[str]] = None,
        bins: int = 10,
    ) -> pd.DataFrame:
        """Get probability densities for reference and current datasets for comparison.

        Args:
            columns: Specific columns to get densities for. If None, all common columns are used.
            bins: Number of bins to use for continuous features.

        Returns:
            DataFrame with density information for each feature and bin, containing:
                - feature: Name of the feature
                - bin: Bin label (category name or numerical range)
                - ref_density: Density in the reference dataset
                - cur_density: Density in the current dataset

        """
        if columns is None:
            columns = list(self._feature_types.keys())
        else:
            columns = [col for col in columns if col in self._feature_types]

        all_densities = []
        for col in columns:
            s1 = self.df1[col].dropna()
            s2 = self.df2[col].dropna()
            col_type = self._feature_types[col]

            if col_type == "categorical":
                ref_counts = s1.value_counts(normalize=True)
                cur_counts = s2.value_counts(normalize=True)
                all_categories = sorted(list(set(ref_counts.index) | set(cur_counts.index)))
                density_df = pd.DataFrame({
                    "bin": all_categories,
                    "ref_density": ref_counts.reindex(all_categories, fill_value=0).values,
                    "cur_density": cur_counts.reindex(all_categories, fill_value=0).values,
                })
            elif col_type == "continuous":
                min_val = min(s1.min(), s2.min())
                max_val = max(s1.max(), s2.max())
                bin_edges = np.linspace(min_val, max_val, bins + 1)

                ref_hist, _ = np.histogram(s1, bins=bin_edges, density=True)
                cur_hist, _ = np.histogram(s2, bins=bin_edges, density=True)

                # Use bin centers or edges for representation
                bin_labels = [f"({bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}]" for i in range(bins)]

                density_df = pd.DataFrame({
                    "bin": bin_labels,
                    "ref_density": ref_hist * np.diff(bin_edges),  # Convert density to probability mass
                    "cur_density": cur_hist * np.diff(bin_edges),
                })
            else:
                continue  # Should not happen if _determine_feature_types is correct

            density_df["feature"] = col
            all_densities.append(density_df[["feature", "bin", "ref_density", "cur_density"]])

        return pd.concat(all_densities, ignore_index=True)

    @classmethod
    def _categorical_drift_calc(cls, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate Cramér's V statistic.

        Args:
            s1: Reference series.
            s2: Current series.

        Returns:
            Cramér's V statistic as a float between 0 (no association) and 1 (perfect association).

        """
        s1_counts = s1.value_counts()
        s2_counts = s2.value_counts()
        all_categories = s1_counts.index.union(s2_counts.index)

        # Create contingency table ensuring all categories are present
        contingency_table = pd.DataFrame({
            "s1": s1_counts.reindex(all_categories, fill_value=0),
            "s2": s2_counts.reindex(all_categories, fill_value=0),
        })

        # Handle edge cases where chi-squared cannot be computed reliably
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            # If both series have the same single category (or are empty), drift is 0
            if s1.nunique() <= 1 and s2.nunique() <= 1:
                # Check if both are empty or have the same single unique value
                s1_unique = s1.unique()
                s2_unique = s2.unique()
                if (s1.empty and s2.empty) or (
                    len(s1_unique) == 1 and len(s2_unique) == 1 and s1_unique[0] == s2_unique[0]
                ):
                    return 0.0
            # Otherwise (different single categories, one empty, etc.), max drift
            return 1.0

        # Calculate chi-squared statistic
        try:
            chi2, _, _, _ = chi2_contingency(contingency_table)
        except ValueError:  # Can happen if a row/column sums to zero after reindexing
            # Check if distributions are effectively identical (e.g., only differ by categories with zero counts)
            norm_s1 = s1_counts / s1_counts.sum()
            norm_s2 = s2_counts / s2_counts.sum()
            merged_counts = pd.DataFrame({"s1": norm_s1, "s2": norm_s2}).fillna(0)
            if merged_counts["s1"].equals(merged_counts["s2"]):
                return 0.0
            else:
                return 1.0  # Assume max drift if chi2 fails and distributions differ

        n = contingency_table.sum().sum()
        if n == 0:
            return 0.0  # No data

        # Calculate phi^2
        phi2 = chi2 / n
        # Number of rows (k) and columns (r)
        k, r = contingency_table.shape

        # Calculate Cramér's V
        min_dim = min(k - 1, r - 1)
        if min_dim == 0:
            # This case should ideally be caught earlier, but handle defensively
            return 0.0

        v = np.sqrt(phi2 / min_dim)
        # Clamp value to [0, 1] due to potential floating point inaccuracies
        return max(0.0, min(1.0, v))

    @classmethod
    def _continuous_drift_calc(cls, s1: pd.Series, s2: pd.Series, bins: int = 10) -> float:
        """Calculate Wasserstein distance (Earth Mover's Distance).

        Args:
            s1: Reference series.
            s2: Current series.
            bins: Number of bins. Not directly used in this implementation but kept for
                 protocol compliance.

        Returns:
            Wasserstein distance between the empirical distributions of s1 and s2.

        """
        # Note: `wasserstein_distance` expects 1D arrays of values, not distributions.
        # It calculates the distance between the empirical distributions.
        # No binning is strictly required by the function itself, but the protocol mentions bins.
        # We will calculate the direct Wasserstein distance between the samples.
        # The `bins` parameter is unused here but kept for protocol compliance signature.
        return wasserstein_distance(s1.values, s2.values)
