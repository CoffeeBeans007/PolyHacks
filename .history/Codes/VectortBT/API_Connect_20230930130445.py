import vectorbt as vbt
import pandas as pd
import numpy as np
from numba import njit
from datetime import datetime, timedelta

class Backtest():
    
    def __init__(self, data, strategy, cash=10000, commission=0.01, margin=1, leverage=1, size=1, slippage=0, log=False):
        self.data = data
        self.strategy = strategy
        self.cash = cash
        self.commission = commission
        self.margin = margin
        self.leverage = leverage
        self.size = size
        self.slippage = slippage
        self.log = log

    def movingAverage(stock_symbol:str, start_date, end_date,reactor:float):
        """
    Calculate portfolio metrics for a given stock symbol within a specified date range.

    Parameters
    ----------
    stock_symbol : str
        The stock symbol or ticker of the stock to analyze.
    start_date : str
        The start date for the analysis in the format 'YYYY-MM-DD UTC'.
    end_date : str
        The end date for the analysis in the format 'YYYY-MM-DD UTC'.
    reactor : float
        Reactor defines how fast the moving average reacts to the price change.
        A higher reactor means a faster moving average.

    Returns
    -------
    float
        The total profit of the portfolio.
    DataFrame
        Portfolio statistics.
    """
        reactor=1-reactor
        if 0<reactor<0.25:
            move1=5
            move2=20
        elif 0.25<=reactor<0.5:
            move1=10
            move2=30
        elif 0.5<=reactor<0.75:
            move1=20
            move2=50
        elif 0.75<=reactor<=1:
            move1=50
            move2=200
        # Download stock data
        
        stockPrice = vbt.YFData.download(stock_symbol, start=start_date, end=end_date).get('Close')
        
        # Calculate moving averages
        fast_ma = vbt.MA.run(stockPrice, move1)
        slow_ma = vbt.MA.run(stockPrice, move2)
        
        # Calculate RSI
        rsi = vbt.RSI.run(stockPrice, 10, short_name='rsi')
        
        # Define entry and exit signals
        entries = fast_ma.ma_crossed_above(slow_ma) & (rsi.rsi_above(50))
        exits = slow_ma.ma_crossed_above(fast_ma) & (rsi.rsi_below(50))
        
        # Create portfolio and calculate metrics
        pf = vbt.Portfolio.from_signals(stockPrice, entries, exits, fees=0.001, freq='d')
        
        return pf.total_profit(), pf.stats()
    


    
    
        