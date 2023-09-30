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

    


    
    
        