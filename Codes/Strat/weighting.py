import numpy as np
import pandas as pd
import re
import cvxpy as cp
from os_helper import OsHelper


class InverseMetricsWeighting(object):
    def __init__(self, filtered_data: pd.DataFrame, years_to_inverse: list[float, int], metric_to_inverse: str,
                 weighting_list: list[float], min_weight: float, max_weight: float):
        self.filtered_data = self._verify_data(filtered_data)

    def _verify_data(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(filtered_data.index, pd.DatetimeIndex):
            try:
                filtered_data.index = pd.to_datetime(filtered_data.index)
                filtered_data.index.name = None
            except ValueError:
                raise ValueError()

if __name__ == "__main__":
    os_helper = OsHelper()
    filtered_data = os_helper.read_data(directory_name="transform data", file_name="filtered_data.csv", index_col=0)
    years_to_inverse = [1, 3]
    print(filtered_data)
    inverse_metrics_weighting = InverseMetricsWeighting(filtered_data=filtered_data)
    print(inverse_metrics_weighting)

    os_helper.write_data(directory_name="transform data", file_name="inverse_metrics_weighting.csv", data_frame=inverse_metrics_weighting)