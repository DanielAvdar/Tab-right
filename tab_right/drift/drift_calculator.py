"""Implementation of the DriftCalcP protocol."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wasserstein_distance


@dataclass
class DriftCalculator:
    """Implementation of DriftCalcP using Cramér's V and Wasserstein distance."""

    df1: pd.DataFrame
    df2: pd.DataFrame
    kind: Optional[Dict[str, str]] = None
    _feature_types: Dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the DriftCalculator with reference and current datasets.

        Raises:
            ValueError: If there are no common columns between the reference and current datasets.

        """
        if not set(self.df1.columns).intersection(set(self.df2.columns)):
            raise ValueError("No common columns between the reference and current datasets.")
        self._feature_types = self._determine_feature_types()

    def _determine_feature_types(self) -> Dict[str, str]:
        """Determine if features are categorical or continuous based on `kind`.

        Returns:
            Dictionary mapping column names to their types ("categorical" or "continuous").

        Raises:
            TypeError: If `kind` is not None or a dict.

        """
        common_cols = list(set(self.df1.columns) & set(self.df2.columns))
        feature_types = {}

        if self.kind is None:
            for col in common_cols:
                # Treat low-cardinality numerics as categorical (<=20 unique values)
                if pd.api.types.is_numeric_dtype(self.df1[col]) and pd.api.types.is_numeric_dtype(self.df2[col]):
                    nunique = min(self.df1[col].nunique(), self.df2[col].nunique())
                    if nunique <= 20:
                        feature_types[col] = "categorical"
                    else:
                        feature_types[col] = "continuous"
                else:
                    feature_types[col] = "categorical"
        elif isinstance(self.kind, dict):
            for col in common_cols:
                t = self.kind.get(col, None)
                if t is not None:
                    feature_types[col] = t
                else:
                    if pd.api.types.is_numeric_dtype(self.df1[col]) and pd.api.types.is_numeric_dtype(self.df2[col]):
                        nunique = min(self.df1[col].nunique(), self.df2[col].nunique())
                        if nunique <= 20:
                            feature_types[col] = "categorical"
                        else:
                            feature_types[col] = "continuous"
                    else:
                        feature_types[col] = "categorical"
        else:
            raise TypeError("`kind` must be None or a dict mapping column names to 'continuous' or 'categorical'.")

        return feature_types

    def __call__(self, columns: Optional[Iterable[str]] = None, bins: int = 10, **kwargs: Any) -> pd.DataFrame:
        """Calculate drift metrics between the reference and current datasets (vectorized).

        Returns
        -------
        pd.DataFrame
            DataFrame with drift metrics for each feature, containing:
            - feature: Name of the feature
            - type: Type of metric used (cramer_v, wasserstein, or N/A)
            - score: Normalized drift score (for Wasserstein, this is the raw score)
            - raw_score: Unnormalized drift metric value

        """
        cols = list(self._feature_types.keys()) if columns is None else [c for c in columns if c in self._feature_types]

        def drift_row(col: str) -> dict:
            s1, s2 = self.df1[col].dropna(), self.df2[col].dropna()
            t = self._feature_types[col]
            if s1.empty or s2.empty:
                return dict(feature=col, type="N/A (Empty Data)", score=np.nan, raw_score=np.nan)
            if t == "categorical":
                v = self._categorical_drift_calc(s1, s2)
                return dict(feature=col, type="cramer_v", score=v, raw_score=v)
            if t == "continuous":
                v = self._continuous_drift_calc(s1, s2, bins=bins)
                return dict(feature=col, type="wasserstein", score=v, raw_score=v)
            raise ValueError(f"Unknown column type '{t}' for column '{col}'")

        return pd.DataFrame([drift_row(col) for col in cols])

    def get_prob_density(self, columns: Optional[Iterable[str]] = None, bins: int = 10) -> pd.DataFrame:
        """Get probability densities for reference and current datasets (vectorized).

        Returns
        -------
        pd.DataFrame
            DataFrame with density information for each feature and bin, containing:
            - feature: Name of the feature
            - bin: Bin label (category name or numerical range)
            - ref_density: Density in the reference dataset
            - cur_density: Density in the current dataset

        """
        cols = list(self._feature_types.keys()) if columns is None else [c for c in columns if c in self._feature_types]
        dens = []
        for col in cols:
            s1, s2 = self.df1[col].dropna(), self.df2[col].dropna()
            t = self._feature_types[col]
            if t == "categorical":
                cats = sorted(set(s1.unique()) | set(s2.unique()))
                ref = s1.value_counts(normalize=True).reindex(cats, fill_value=0)
                cur = s2.value_counts(normalize=True).reindex(cats, fill_value=0)
                d = pd.DataFrame({"bin": cats, "ref_density": ref.values, "cur_density": cur.values})
            elif t == "continuous":
                minv, maxv = min(s1.min(), s2.min()), max(s1.max(), s2.max())
                edges = np.linspace(minv, maxv, bins + 1)
                ref_hist, _ = np.histogram(s1, bins=edges, density=True)
                cur_hist, _ = np.histogram(s2, bins=edges, density=True)
                labels = [f"({edges[i]:.2f}-{edges[i + 1]:.2f}]" for i in range(bins)]
                d = pd.DataFrame({
                    "bin": labels,
                    "ref_density": ref_hist * np.diff(edges),
                    "cur_density": cur_hist * np.diff(edges),
                })
            else:
                continue
            d["feature"] = col
            dens.append(d[["feature", "bin", "ref_density", "cur_density"]])
        return pd.concat(dens, ignore_index=True)

    @classmethod
    def _categorical_drift_calc(cls, s1: pd.Series, s2: pd.Series) -> float:
        """Simplified Cramér's V statistic calculation.

        Args:
            s1: Reference series.
            s2: Current series.

        Returns:
            float: Cramér's V statistic between 0 (no drift) and 1 (max drift).

        """
        # Quick check for identical or empty series
        if s1.equals(s2) or (s1.empty and s2.empty):
            return 0.0

        # Create raw count distributions
        s1_counts = s1.value_counts()
        s2_counts = s2.value_counts()
        all_cats = sorted(set(s1_counts.index) | set(s2_counts.index))
        table = pd.DataFrame({
            "s1": s1_counts.reindex(all_cats, fill_value=0),
            "s2": s2_counts.reindex(all_cats, fill_value=0),
        })

        # Handle edge cases where chi-squared calculation would fail
        if len(all_cats) < 2 or table.shape[1] < 2:
            return 0.0 if table["s1"].equals(table["s2"]) else 1.0

        try:
            chi2, _, _, _ = chi2_contingency(table)
            n = table.values.sum()
            min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
            v = np.sqrt(chi2 / (n * min_dim))
            return max(0.0, min(1.0, v))
        except ValueError:
            return 0.0 if table["s1"].equals(table["s2"]) else 1.0

    @classmethod
    def _continuous_drift_calc(cls, s1: pd.Series, s2: pd.Series, bins: int = 10) -> float:
        """Simplified Wasserstein distance calculation.

        Args:
            s1: Reference series.
            s2: Current series.
            bins: Number of bins (not used, for protocol compatibility).

        Returns:
            float: Wasserstein distance between the two distributions.
                Returns 0.0 if both are empty, 1.0 if only one is empty.

        """
        if s1.empty or s2.empty:
            return 0.0 if s1.empty and s2.empty else 1.0
        return wasserstein_distance(s1.values, s2.values)
