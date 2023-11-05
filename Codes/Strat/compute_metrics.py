import pandas as pd
import numpy as np
import datetime
from os_helper import OsHelper

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)


class ComputeMetrics(object):
    def __init__(self, price_data: pd.DataFrame, trade_volume: pd.DataFrame, reference_index: str, rolling_window_years: list = None):
        self.price_data = self._validate_data(price_data)
        self.trade_volume = self._validate_data(trade_volume)
        self.reference_index = self._verify_index(reference_index)
        self.rolling_window_years = rolling_window_years
        self.turnover = self._compute_turnover()
        self.index_returns = self._get_index_returns()
        self.returns = self._compute_returns()

    @staticmethod
    def _verify_index(reference_index: str) -> str:
        if reference_index not in price_data.columns:
            raise ValueError(f"Reference index {reference_index} not found in price filtered_data.")
        return reference_index

    @staticmethod
    def _validate_datetime_format(dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.index.dtype == "O":
            dataset.index = pd.to_datetime(dataset.index)
        return dataset

    @staticmethod
    def _impute_missing_data(dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.replace(0, np.nan, inplace=True)

        for ticker in dataset.columns:
            col_series = dataset[ticker]
            first_valid_idx = col_series.first_valid_index()
            last_valid_idx = col_series.last_valid_index()

            if first_valid_idx is not None and last_valid_idx is not None:
                # Forward fill only between the first and last non-NaN values
                dataset[ticker].loc[first_valid_idx:last_valid_idx] = \
                    dataset[ticker].loc[first_valid_idx:last_valid_idx].ffill()

        return dataset

    @staticmethod
    def _drop_irregular_columns(dataset: pd.DataFrame) -> pd.DataFrame:
        initial_col_count = dataset.shape[1]
        valid_cols = [col for col in dataset.columns if
                      all(isinstance(x, (float, int)) or pd.isna(x) for x in dataset[col])]
        dropped_cols = set(dataset.columns) - set(valid_cols)

        dataset = dataset[valid_cols]
        final_col_count = dataset.shape[1]

        print(f"Dropped columns: {dropped_cols}")
        print(f"Number of columns before: {initial_col_count}")
        print(f"Number of columns after: {final_col_count}")

        return dataset

    def _validate_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = self._validate_datetime_format(dataset)
        dataset = self._drop_irregular_columns(dataset)
        dataset = dataset.sort_index()
        return dataset

    def _compute_turnover(self) -> pd.DataFrame:
        return self.trade_volume * self.price_data

    def _compute_returns(self) -> pd.DataFrame:
        returns = self.price_data.pct_change(fill_method=None).iloc[1:, :]
        returns.drop(labels=[self.reference_index], axis=1, inplace=True)
        return returns

    def _get_index_returns(self) -> pd.Series:
        index_series = self.price_data[self.reference_index].copy()
        index_returns = index_series.pct_change(fill_method=None).iloc[1:]
        return index_returns

    def compute_average_rolling_turnover(self, rolling_window_year: int) -> pd.DataFrame:
        rolling_window = int(rolling_window_year * 252)
        return self.turnover.rolling(window=rolling_window).mean()

    def compute_rolling_volatility(self, rolling_window_year: int) -> pd.DataFrame:
        rolling_window = int(rolling_window_year * 252)
        return self.returns.rolling(window=rolling_window).std() * np.sqrt(252)

    def compute_rolling_beta(self, rolling_window_year: int) -> pd.DataFrame:
        rolling_window = int(rolling_window_year * 252)
        # Compute the rolling covariance and variance
        covariance_with_index = self.returns.rolling(window=rolling_window).cov(self.index_returns, pairwise=True)
        variance_of_index = self.index_returns.rolling(window=rolling_window).var()

        # Calculate rolling betas
        rolling_betas = covariance_with_index.div(variance_of_index.values, axis=0)

        return rolling_betas

    def compile_all_metrics(self) -> pd.DataFrame:

        all_risk_metrics_dict = {ticker: {} for ticker in self.returns.columns}

        start = datetime.datetime.now()
        print(f'all metrics calculation started {start.strftime("%Y-%m-%d %H:%M:%S")}')

        for year in self.rolling_window_years:
            for ticker in self.returns.columns:
                # Average rolling turnover
                average_rolling_turnover = self.compute_average_rolling_turnover(year)
                all_risk_metrics_dict[ticker][f"average_turnover_{year}Y"] = average_rolling_turnover[ticker]
                # Volatility
                volatility = self.compute_rolling_volatility(year)
                all_risk_metrics_dict[ticker][f"volatility_{year}Y"] = volatility[ticker]

                # Beta
                beta = self.compute_rolling_beta(year)
                all_risk_metrics_dict[ticker][f"beta_{year}Y"] = beta[ticker]

            print(f"Metrics for {year} year(s) window calculated")

        all_risk_metrics_df = pd.concat(
            {k: pd.DataFrame(v) for k, v in all_risk_metrics_dict.items()}, axis=1
        )
        all_risk_metrics_df = all_risk_metrics_df.sort_index(axis=1, level=[0, 1])

        end = datetime.datetime.now()
        print(f'all metrics calculation ended {end.strftime("%Y-%m-%d %H:%M:%S")}')
        duration = end - start
        duration_in_minutes = duration.total_seconds() / 60
        print(f"duration for all metrics calculation {duration_in_minutes:.4f} minutes")

        return all_risk_metrics_df



if __name__ == '__main__':
    os_helper = OsHelper()

    price_data = os_helper.read_data(directory_name="base filtered_data", file_name="previous_close.csv", index_col=0, low_memory=False)
    trade_volume = os_helper.read_data(directory_name="base filtered_data", file_name="previous_volume.csv", index_col=0, low_memory=False)

    rolling_window_years = [1, 3]

    metrics_calculator = ComputeMetrics(price_data=price_data, trade_volume=trade_volume, rolling_window_years=rolling_window_years,
                                        reference_index="AAPL US Equity")

    final_df = metrics_calculator.compile_all_metrics()

    os_helper.write_data(directory_name="transform filtered_data", file_name="all_metrics.csv", data_frame=final_df)





