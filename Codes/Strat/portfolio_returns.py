import pandas as pd
import numpy as np
from os_helper import OsHelper


class PortfolioReturns(object):
    def __init__(self, prices: pd.DataFrame, weighting_data: pd.DataFrame, rebalance_lag: int = 7):
        self.prices = self._validate_price_data(prices)
        self.rebalance_lag = rebalance_lag
        self.asset_returns = self._compute_returns()
        self.weighting_data = self._transform_allocation_data(weighting_data)
        self.drifted_weights = self._compute_drifted_weights()
        self.portfolio_returns = self.compute_portfolio_returns()

    def _validate_price_data(self, asset_prices: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            asset_prices.index = pd.to_datetime(asset_prices.index)
            asset_prices.index.name = None
        return asset_prices

    def _transform_allocation_data(self, allocation_data: pd.DataFrame) -> pd.DataFrame:
        allocation_data['Date'] = pd.to_datetime(allocation_data['Date'])
        reshaped_data = allocation_data.pivot(index='Date', columns='Ticker', values='Weight')
        return reshaped_data

    def _impute_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.replace(0, np.nan, inplace=True)
        for ticker in dataset.columns:
            col_series = dataset[ticker]
            first_valid_idx = col_series.first_valid_index()
            last_valid_idx = col_series.last_valid_index()
            if first_valid_idx and last_valid_idx:
                dataset[ticker].loc[first_valid_idx:last_valid_idx] = \
                    dataset[ticker].loc[first_valid_idx:last_valid_idx].ffill()
        return dataset

    def _compute_returns(self) -> pd.DataFrame:
        self.prices = self._impute_data(dataset=self.prices)
        returns = self.prices.pct_change(fill_method=None).iloc[1:, :]
        returns.index.name = None
        return returns

    def _compute_drifted_weights(self) -> pd.DataFrame:
        allocation = self.weighting_data.copy()
        returns = self.asset_returns.copy()
        allocation, returns = allocation.align(returns, axis=1, join="inner")
        allocation.index = allocation.index.shift(self.rebalance_lag, freq='D')
        allocation.fillna(0, inplace=True)
        first_rebalance_date = allocation.index[0]
        returns = returns.loc[first_rebalance_date:]
        adjusted_weights = pd.DataFrame(index=returns.index, columns=allocation.columns)
        current_weights = allocation.loc[first_rebalance_date]
        adjusted_weights.loc[first_rebalance_date] = current_weights
        for i in range(1, len(returns)):
            if returns.index[i] in allocation.index:
                current_weights = allocation.loc[returns.index[i]]
            else:
                daily_change = current_weights * (1 + returns.iloc[i])
                daily_change[returns.iloc[i].isna()] = 0
                current_weights = daily_change / daily_change.sum()
            adjusted_weights.iloc[i] = current_weights
        adjusted_weights.index.name = None
        return adjusted_weights

    def compute_portfolio_returns(self) -> pd.DataFrame:
        returns = self.asset_returns.loc[self.drifted_weights.index[0]:]
        portfolio_returns = pd.DataFrame(index=returns.index, columns=["Portfolio_Returns"])
        portfolio_returns["Portfolio_Returns"] = (self.drifted_weights.shift(periods=1) * returns).sum(axis=1)
        portfolio_returns.index.name = None
        return portfolio_returns




if __name__ == "__main__":
    os_helper = OsHelper()
    prices = os_helper.read_data(directory_name="base data", file_name="previous_adjusted_close.csv", index_col=0)
    print(prices.head())
    weighting_df = os_helper.read_data(directory_name="transform data", file_name="inverse_metrics_weighting.csv", index_col=0)
    print(weighting_df.head())

    rebalance_lag = 7

    pf_returns = PortfolioReturns(prices=prices, weighting_data=weighting_df, rebalance_lag=rebalance_lag)
    drifted_weights = pf_returns.drifted_weights
    print(drifted_weights.head())
    portfolio_returns = pf_returns.portfolio_returns
    print(portfolio_returns.head())

    os_helper.write_data(directory_name="final data", file_name="portfolio_returns.csv", data_frame=portfolio_returns)
    os_helper.write_data(directory_name="final data", file_name="drifted_weights.csv", data_frame=drifted_weights)






