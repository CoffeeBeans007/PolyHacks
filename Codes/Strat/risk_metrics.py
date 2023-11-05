import pandas as pd
import datetime
import os
import numpy as np
from typing import Union, Set, Any
import math
from file_management import FileManagement

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

class RiskMetrics(object):
    def __init__(self, close_data: pd.DataFrame, volume_data: pd.DataFrame, benchmark_name: str, frequency: str = "D", years_metrics_list: list = None,
                 weekday_resampling: str = "FRI"):

        self.close_data = self._verify_data(close_data)
        self.volume_data = self._verify_data(volume_data)
        self.benchmark_name = benchmark_name
        self.frequency = self._verify_frequency(frequency)
        self.years_metrics_list = self._verify_years_metrics_list(years_metrics_list)
        self.weekday_resampling = self._verify_weekday_resampling(weekday_resampling)

        def _verify_datetime_format(data: pd.DataFrame) -> pd.DataFrame:
            if data.index.dtype == "O":
                data.index = pd.to_datetime(data.index)
            return data

        def _verify_data(data: pd.DataFrame) -> pd.DataFrame:
            data = _verify_datetime_format(data)
            data = data.sort_index()
            return data



if __name__ == '__main__':
    fm = FileManagement()

    close_data = fm.load_data(folder_name="Data", file_name="previous_close.csv", index_col=0, low_memory=False)
    volume_data = fm.load_data(folder_name="Data", file_name="previous_volume.csv", index_col=0, low_memory=False)

    print(close_data.head())
    print(volume_data.head())




