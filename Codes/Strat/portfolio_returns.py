import pandas as pd
import numpy as np
from typing import Dict, Tuple
from os_helper import OsHelper


pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)


class PortfolioReturns(object):
    def __init__(self, prices: pd.DataFrame, weighting_data: pd.DataFrame, rebalance_lag: int = 7, transaction_fee: float = 0.001):
        """
        Initializes the PortfolioReturns class.

        Args:
            prices (pd.DataFrame): A DataFrame containing asset prices.
            weighting_data (pd.DataFrame): A DataFrame containing asset allocation data.
            rebalance_lag (int, optional): The rebalance lag. Defaults to 7.
        """
        # Validate and prepare data for computations
        self.prices = self._validate_price_data(prices)
        self.transaction_fee = transaction_fee
        self.rebalance_lag = rebalance_lag
        self.weighting_df = weighting_data
        self.asset_returns = self._compute_returns()
        self.weighting_data = self._transform_allocation_data(weighting_data)
        self.drifted_weights = self._compute_drifted_weights()
        self.portfolio_returns = self.compute_portfolio_returns()

    def _validate_price_data(self, asset_prices: pd.DataFrame, cutoff_date: str = '2015-01-01') -> pd.DataFrame:
        """
        Validates and transforms the price data, keeping only the data up to the specified cutoff date.

        Args:
            asset_prices (pd.DataFrame): A DataFrame containing asset prices.
            cutoff_date (str): The cutoff date in 'YYYY-MM-DD' format, up to which data should be retained.

        Returns:
            pd.DataFrame: Transformed and filtered asset prices DataFrame.
        """
        # Ensure the index is a datetime
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            asset_prices.index = pd.to_datetime(asset_prices.index)

        # Filter the DataFrame to include only dates up to the cutoff
        cutoff_datetime = pd.to_datetime(cutoff_date)
        filtered_prices = asset_prices[asset_prices.index <= cutoff_datetime]

        return filtered_prices

    def _transform_allocation_data(self, allocation_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms allocation data into the desired format.

        Args:
            allocation_data (pd.DataFrame): A DataFrame containing allocation data.

        Returns:
            pd.DataFrame: Transformed allocation data.
        """
        # Pivot the allocation data for easier computations
        allocation_data['Date'] = pd.to_datetime(allocation_data['Date'])
        reshaped_data = allocation_data.pivot(index='Date', columns='Ticker', values='Weight')
        return reshaped_data

    def _impute_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing data in a DataFrame.

        Args:
            dataset (pd.DataFrame): A DataFrame with potentially missing data.

        Returns:
            pd.DataFrame: DataFrame with imputed data.
        """
        # Replace 0 with NaN for better handling of missing data
        dataset.replace(0, np.nan, inplace=True)
        # Forward-fill missing data for each ticker
        for ticker in dataset.columns:
            col_series = dataset[ticker]
            first_valid_idx = col_series.first_valid_index()
            last_valid_idx = col_series.last_valid_index()
            if first_valid_idx and last_valid_idx:
                dataset[ticker].loc[first_valid_idx:last_valid_idx] = \
                    dataset[ticker].loc[first_valid_idx:last_valid_idx].ffill()
        return dataset

    def _compute_returns(self) -> pd.DataFrame:
        """
        Computes asset returns based on prices.

        Returns:
            pd.DataFrame: A DataFrame containing asset returns.
        """
        # Impute missing data before computing returns
        self.prices = self._impute_data(dataset=self.prices)
        # Compute percentage change
        returns = self.prices.pct_change(fill_method=None).iloc[1:, :]
        returns.index.name = None
        return returns

    def verify_and_normalize_weights(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Verifies if the weights on each date sum to 1. If not, normalizes them.

        Args:
            weights_df (pd.DataFrame): DataFrame with weights to verify.

        Returns:
            pd.DataFrame: The DataFrame with weights normalized if necessary.
        """
        for date, weights in weights_df.iterrows():
            weights_sum = weights.sum()
            # If the sum of weights is not 1, normalize the weights
            if not np.isclose(weights_sum, 1):
                print(f"Sum of weights on {date} is not 1. Normalizing weights.")
                weights_df.loc[date] = weights / weights_sum
        return weights_df

    def _compute_drifted_weights(self) -> pd.DataFrame:
        """
        Computes drifted weights based on initial allocations and asset returns.

        Returns:
            pd.DataFrame: A DataFrame containing drifted weights.
        """
        # Align data for computation
        allocation = self.weighting_data.copy()
        returns = self.asset_returns.copy()
        allocation, returns = allocation.align(returns, axis=1, join="inner")

        # Adjust dates for rebalance lag
        allocation.index = allocation.index.shift(self.rebalance_lag, freq='D')
        allocation.fillna(0, inplace=True)

        # Initialize drifted weights
        first_rebalance_date = allocation.index[0]
        returns = returns.loc[first_rebalance_date:]
        adjusted_weights = pd.DataFrame(index=returns.index, columns=allocation.columns)
        current_weights = allocation.loc[first_rebalance_date]
        adjusted_weights.loc[first_rebalance_date] = current_weights

        # Compute drifted weights
        for i in range(1, len(returns)):
            if returns.index[i] in allocation.index:
                current_weights = allocation.loc[returns.index[i]]
            else:
                daily_change = current_weights * (1 + returns.iloc[i])
                daily_change[returns.iloc[i].isna()] = 0
                current_weights = daily_change / daily_change.sum()
            adjusted_weights.iloc[i] = current_weights

        # Verify and normalize drifted weights
        adjusted_weights = self.verify_and_normalize_weights(adjusted_weights)

        adjusted_weights.index.name = None
        return adjusted_weights

    def _create_sector_mapping(self) -> Dict[str, str]:
        """
        Creates a mapping from tickers to sectors using the weighting data.

        Returns:
            Dict[str, str]: A dictionary mapping tickers to their respective sectors.
        """
        sector_mapping = self.weighting_df[['Ticker', 'Sector']].drop_duplicates().set_index('Ticker')[
            'Sector'].to_dict()
        return sector_mapping

    def compute_sectors_returns(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes the drifted weights and compounded returns of sectors by mapping tickers to sectors and summing the weights.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing a DataFrame of sector-level drifted weights and a DataFrame of sector-level compounded returns.
        """

        sector_mapping = self._create_sector_mapping()

        # Map each ticker in the drifted weights to its sector
        sector_weights = self.drifted_weights.rename(columns=sector_mapping)

        # Group by sector and sum the weights
        sector_weights = sector_weights.groupby(by=sector_weights.columns, axis=1).sum()
        sector_weights.dropna(how='all', inplace=True)

        # Compute sector-level returns
        sector_returns = sector_weights.shift(periods=1) * self.asset_returns.groupby(by=sector_mapping, axis=1).sum()
        # Compute the compounded returns
        sectors_compounded_returns = (1 + sector_returns).cumprod()
        # Drop lines where NaN are in all columns
        sectors_compounded_returns.dropna(how='all', inplace=True)

        return sector_weights, sectors_compounded_returns

    def compute_portfolio_returns(self) -> pd.DataFrame:
        """
        Computes the overall portfolio returns based on drifted weights and asset returns.

        Returns:
            pd.DataFrame: A DataFrame containing the portfolio returns.
        """
        # Calculate portfolio returns by multiplying drifted weights with asset returns
        returns = self.asset_returns.loc[self.drifted_weights.index[0]:]
        portfolio_returns = pd.DataFrame(index=returns.index, columns=["Portfolio_Returns"])
        portfolio_returns["Portfolio_Returns"] = (self.drifted_weights.shift(periods=1) * returns).sum(axis=1)

        # Initialize a series to store transaction costs, defaulting to 0
        transaction_costs = pd.Series(0, index=portfolio_returns.index)

        # Adjust for transaction costs on rebalance days
        rebalance_dates = self.weighting_data.index.shift(self.rebalance_lag, freq='D')

        for date in rebalance_dates:
            try:
                # Attempt to calculate transaction costs
                transaction_cost = self.transaction_fee * np.abs(self.drifted_weights.loc[date] - self.drifted_weights.shift(periods=1).loc[date]).sum()
                transaction_costs.loc[date] = transaction_cost
            except KeyError:
                # If a KeyError occurs, set transaction cost to 0 for this date
                transaction_costs.loc[date] = 0
                print(f"Unable to calculate transaction costs for {date} (Holiday).")

            # Subtract transaction costs from portfolio returns
        portfolio_returns["Portfolio_Returns"] -= transaction_costs

        portfolio_returns.index.name = None
        return portfolio_returns


if __name__ == "__main__":
    os_helper = OsHelper()
    prices = os_helper.read_data(directory_name="base data", file_name="new_tot_ret.csv", index_col=0, sep=',')
    print(prices.head())
    weighting_df = os_helper.read_data(directory_name="transform data", file_name="inverse_metrics_weighting.csv", index_col=0)
    print(weighting_df.head())

    rebalance_lag = 7
    transaction_fee = 0.001

    pf_returns = PortfolioReturns(prices=prices, weighting_data=weighting_df, rebalance_lag=rebalance_lag, transaction_fee=transaction_fee)
    drifted_weights = pf_returns.drifted_weights
    print(drifted_weights.head())
    portfolio_returns = pf_returns.portfolio_returns
    print(portfolio_returns.head())

    sectors_drifted_weights, sectors_compounded_returns = pf_returns.compute_sectors_returns()
    print(sectors_drifted_weights.head())
    print(sectors_compounded_returns.head())

    os_helper.write_data(directory_name="final data", file_name="portfolio_returns.csv", data_frame=portfolio_returns)
    os_helper.write_data(directory_name="final data", file_name="drifted_weights.csv", data_frame=drifted_weights)
    os_helper.write_data(directory_name="final data", file_name="sectors_drifted_weights.csv", data_frame=sectors_drifted_weights)
    os_helper.write_data(directory_name="final data", file_name="sectors_compounded_returns.csv", data_frame=sectors_compounded_returns)






