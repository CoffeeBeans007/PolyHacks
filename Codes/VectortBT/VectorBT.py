#########
# Main class implementation for vectorBT pro backtester
# Created by: Coffeebeans007
#########


import vectorbtpro as vbt
import numpy as np
import pandas as pd
import numba
from numba import njit
import datetime
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.optimize as spop
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime as daee
from itertools import combinations
import json
import requests



class BackTester:
    """
    Main class for vectorBT pro backtester
    """
    def __init__(self,date,) -> None:
        pass

    def get_data(self, symbol, start_date, end_date):
        """
        Get data from yahoo finance 
        """
        #TODO add more data sources
        return data
    
    def CreateIndicator(self, indicator,dataset,strategy,params,pfValue,fees):
        """
        Create an indicator
        """
        #TODO add more indicators
        indicator=vbt.IndicatorFactory(
            class_name=strategy,
            short_name=strategy,
            input_name=[params[0]],
            param_names=params[1],
            output_names=["signal"],
        ).with_apply_func(
            strategy,
        )
        results=indicator.run(dataset)
        entries= results.signal ==1
        exits= results.signal ==-1

        portfolio = vbt.Portfolio.from_signals(dataset,entries,exits,fees=fees, init_cash=pfValue)
        return portfolio