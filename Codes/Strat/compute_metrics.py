import pandas as pd
import numpy as np
import datetime
from os_helper import OsHelper

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)


class ComputeMetrics(object):
    def __init__(self, price_data: pd.DataFrame, trade_volume: pd.DataFrame, benchmark_series: pd.Series, rolling_window_years: list = None):
        """
        Initialize ComputeMetrics with price data, trade volume, reference index, and rolling window years.

        Args:
            price_data (pd.DataFrame): DataFrame containing historical price data.
            trade_volume (pd.DataFrame): DataFrame containing trade volume data.
            reference_index (str): The reference index for calculating metrics.
            rolling_window_years (list, optional): List of rolling window periods in years. Defaults to None.
        """
        self.price_data = self._validate_data(price_data)
        self.trade_volume = self._validate_data(trade_volume)
        self.rolling_window_years = rolling_window_years
        self.turnover = self._compute_turnover()
        self.returns = self._compute_returns()
        self.benchmark_returns = self._prepare_benchmark_returns(benchmark_series)


    @staticmethod
    def _validate_datetime_format(dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the datetime format of the index of the dataset.
        Args:
            dataset: Dataset to validate.

        Returns:
            pd.DataFrame: DataFrame with validated datetime format.
        """
        if dataset.index.dtype == "O":
            dataset.index = pd.to_datetime(dataset.index)
        return dataset

    @staticmethod
    def _impute_missing_data(dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing data by forward filling.

        Returns:
            pd.DataFrame: DataFrame with imputed missing data.
        """
        dataset.replace(0, np.nan, inplace=True)

        for ticker in dataset.columns:
            col_series = dataset[ticker]
            first_valid_idx = col_series.first_valid_index()
            last_valid_idx = col_series.last_valid_index()

            if first_valid_idx is not None and last_valid_idx is not None:
                # Forward fill missing data
                dataset[ticker].loc[first_valid_idx:last_valid_idx] = \
                    dataset[ticker].loc[first_valid_idx:last_valid_idx].ffill()

        return dataset

    @staticmethod
    def _drop_irregular_columns(dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that are not float or int.
        Args:
            dataset: Dataset to drop columns from.

        Returns:
            pd.DataFrame: DataFrame with dropped columns.
        """
        initial_col_count = dataset.shape[1]
        # Drop columns that are not float or int
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
        """
        Validate the dataset by ensuring the datetime format of the index, imputing missing data, and dropping irregular columns.
        Args:
            dataset: Dataset to validate.

        Returns:
            pd.DataFrame: DataFrame with validated dataset.
        """
        dataset = self._validate_datetime_format(dataset)
        dataset = self._drop_irregular_columns(dataset)
        dataset = dataset.sort_index()
        return dataset

    def _prepare_benchmark_returns(self, benchmark_series: pd.Series) -> pd.Series:
        """
        Prepare benchmark returns by calculating daily returns and removing the first row.

        Args:
            benchmark_series (pd.Series): Series containing benchmark data.

        Returns:
            pd.Series: Series containing benchmark returns.
        """
        if benchmark_series.index.dtype == "O":
            benchmark_series.index = pd.to_datetime(benchmark_series.index)
        benchmark_returns = benchmark_series.pct_change(fill_method=None).iloc[1:]
        # keep same index as price data
        benchmark_returns = benchmark_returns.reindex(self.returns.index)
        return benchmark_returns

    def _compute_turnover(self) -> pd.DataFrame:
        """
        Compute turnover as product of trade volume and price data.

        Returns:
            pd.DataFrame: DataFrame of turnover values.
        """
        return self.trade_volume * self.price_data

    def _compute_returns(self) -> pd.DataFrame:
        returns = self.price_data.pct_change(fill_method=None).iloc[1:, :]
        mask = self.price_data.isna()
        returns[mask] = np.nan
        return returns

    def compute_average_rolling_turnover(self, rolling_window_year: int) -> pd.DataFrame:
        """
        Compute average rolling turnover for a given window.

        Args:
            rolling_window_year (int): Rolling window size in years.

        Returns:
            pd.DataFrame: DataFrame of average rolling turnover.
        """
        rolling_window = int(rolling_window_year * 252)
        return self.turnover.rolling(window=rolling_window).mean()

    def compute_rolling_volatility(self, rolling_window_year: int) -> pd.DataFrame:
        """
        Compute rolling volatility for a given window.

        Args:
            rolling_window_year (int): Rolling window size in years.

        Returns:
            pd.DataFrame: DataFrame of rolling volatility.
        """
        rolling_window = int(rolling_window_year * 252)
        return self.returns.rolling(window=rolling_window).std() * np.sqrt(252)

    def compute_rolling_beta(self, rolling_window_year: int) -> pd.DataFrame:
        """
        Compute rolling beta for a given window.

        Args:
            rolling_window_year (int): Rolling window size in years.

        Returns:
            pd.DataFrame: DataFrame of rolling betas.
        """
        rolling_window = int(rolling_window_year * 252)
        # Compute the rolling covariance and variance
        covariance_with_index = self.returns.rolling(window=rolling_window).cov(self.benchmark_returns, pairwise=True)
        variance_of_index = self.benchmark_returns.rolling(window=rolling_window).var()

        # Calculate rolling betas
        rolling_betas = covariance_with_index.div(variance_of_index.values, axis=0)

        return rolling_betas

    def compile_all_metrics(self) -> pd.DataFrame:
        """
        Compile and calculate multiple risk metrics for each ticker.

        This function iterates over different rolling windows specified in self.rolling_window_years,
        and for each window, computes the average rolling turnover, rolling volatility, and rolling beta
        for each ticker in self.returns.columns.

        The function consolidates all calculated metrics into a single DataFrame and returns it.

        Returns:
            pd.DataFrame: A DataFrame containing average turnover, volatility, and beta for each ticker
                          and each rolling window.
        """

        all_metrics_dict = {ticker: {} for ticker in self.returns.columns}

        start = datetime.datetime.now()
        print(f'all metrics calculation started {start.strftime("%Y-%m-%d %H:%M:%S")}')

        for year in self.rolling_window_years:
            for ticker in self.returns.columns:
                # Average rolling turnover
                average_rolling_turnover = self.compute_average_rolling_turnover(year)
                all_metrics_dict[ticker][f"average_turnover_{year}Y"] = average_rolling_turnover[ticker]
                # Volatility
                volatility = self.compute_rolling_volatility(year)
                all_metrics_dict[ticker][f"volatility_{year}Y"] = volatility[ticker]
                # Beta
                beta = self.compute_rolling_beta(year)
                all_metrics_dict[ticker][f"beta_{year}Y"] = beta[ticker]

            print(f"Metrics for {year} year(s) window calculated")

        all_metrics_df = pd.concat(
            {k: pd.DataFrame(v) for k, v in all_metrics_dict.items()}, axis=1
        )
        all_metrics_df = all_metrics_df.sort_index(axis=1, level=[0, 1])

        end = datetime.datetime.now()
        print(f'all metrics calculation ended {end.strftime("%Y-%m-%d %H:%M:%S")}')
        duration = end - start
        duration_in_minutes = duration.total_seconds() / 60
        print(f"duration for all metrics calculation {duration_in_minutes:.4f} minutes")

        return all_metrics_df


if __name__ == '__main__':
    os_helper = OsHelper()

    price_data = os_helper.read_data(directory_name="base data", file_name="close_fixed.csv", index_col=0, low_memory=False)
    trade_volume = os_helper.read_data(directory_name="base data", file_name="volume_fixed.csv", index_col=0, low_memory=False)
    spy = os_helper.read_data(directory_name="benchmark", file_name="SPY (1).csv", index_col=0, low_memory=False)

    print(spy)

    rolling_window_years = [1, 3]

    metrics_calculator = ComputeMetrics(price_data=price_data, trade_volume=trade_volume, rolling_window_years=rolling_window_years,
                                        benchmark_series=spy['Adj Close'])

    final_df = metrics_calculator.compile_all_metrics()

    os_helper.write_data(directory_name="transform data", file_name="all_metrics.csv", data_frame=final_df)





