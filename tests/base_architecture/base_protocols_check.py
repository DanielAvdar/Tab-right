from typing import Any, Callable

import pandas as pd


class CheckProtocols:
    """Base class for checking protocol compliance."""

    class_to_check: Any = None

    def get_metric(self, agg: bool = False) -> Callable:
        def metric_single(y: pd.Series, p: pd.Series) -> pd.Series:
            return abs(y - p)

        def agg_metric_single(y: pd.Series, p: pd.Series) -> float:
            return float(abs(y - p).mean())

        if agg:
            return agg_metric_single
        return metric_single

    def test_protocol_followed(self, instance_to_check: Any) -> None:
        """Test if the protocol is followed correctly."""
        # check if it is dataclass
        assert hasattr(instance_to_check, "__dataclass_fields__")
        assert isinstance(instance_to_check, self.class_to_check)
