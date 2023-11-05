import pandas as pd
import numpy as np
from os_helper import OsHelper

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

class ComputeMetrics(object):
    def __init__(self, price_data: pd.DataFrame, trade_volume: pd.DataFrame, reference_index: str, rolling_window_list: list = None):
        self.price_data = self._validate_data(price_data)
        self.trade_volume = self._validate_data(trade_volume)
        self.reference_index = reference_index
        self.rolling_window_list = rolling_window_list

    @staticmethod
    def _validate_datetime_format(dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.index.dtype == "O":
            dataset.index = pd.to_datetime(dataset.index)
        return dataset

    @staticmethod
    def _impute_missing_data(dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.replace(0, np.nan, inplace=True)
        dataset.interpolate(method='time', inplace=True)
        return dataset

    def _validate_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = self._validate_datetime_format(dataset)
        dataset = dataset.sort_index()
        return dataset


if __name__ == '__main__':
    file_manager = OsHelper()

    price_data = file_manager.read_data(directory_name="Data", file_name="previous_close.csv", index_col=0, low_memory=False)
    trade_volume = file_manager.read_data(directory_name="Data", file_name="previous_volume.csv", index_col=0, low_memory=False)

    metrics_calculator = ComputeMetrics(price_data=price_data, trade_volume=trade_volume,
                                        reference_index="AAPL US Equity")

    metrics_calculator._validate_data(price_data)

    print(price_data.head())
    print(trade_volume.head())




