import pandas as pd
from typing import Union

class Weights:
    """
    A class to represent portfolio weights and to calculate the necessary trades
    to reach the goal weights from the current weights.

    Attributes:
        current_weights (pd.DataFrame): DataFrame with current portfolio weights.
        goal_weights (pd.DataFrame): DataFrame with target portfolio weights.
        prices (pd.DataFrame): DataFrame with the prices of portfolio stocks.
        returns (pd.DataFrame): DataFrame with the daily returns of the stocks.
    """

    def __init__(self, current_weights: pd.DataFrame, goal_weights: pd.DataFrame, 
                 prices: pd.DataFrame, returns: pd.DataFrame):
        """
        Constructs all the necessary attributes for the Weights object.

        Parameters:
            current_weights (pd.DataFrame): DataFrame with current portfolio weights.
            goal_weights (pd.DataFrame): DataFrame with target portfolio weights.
            prices (pd.DataFrame): DataFrame with the prices of portfolio stocks.
            returns (pd.DataFrame): DataFrame with the daily returns of the stocks.
        """
        self.current_weights = current_weights
        self.goal_weights = goal_weights
        self.prices = prices
        self.returns = returns

    def calculate_trades(self, total_portfolio_value: Union[float, int]) -> pd.DataFrame:
        """
        Calculate the number of shares to trade for each stock to reach the goal weights.

        Parameters:
            total_portfolio_value (float or int): The total value of the portfolio.

        Returns:
            pd.DataFrame: A DataFrame with the same index and columns as `current_weights`
                          and `goal_weights` that contains the number of shares to be traded
                          for each stock to reach the goal weight.
        """
        # Calculate the current value of each stock
        current_value = self.current_weights * total_portfolio_value
        
        # Calculate the goal value of each stock
        goal_value = self.goal_weights * total_portfolio_value
        
        # Calculate the difference in value for each stock to reach the goal
        value_difference = goal_value - current_value
        
        # Calculate the number of shares to trade for each stock using the most recent prices
        trades = value_difference.div(self.prices.iloc[-1])
        
        return trades