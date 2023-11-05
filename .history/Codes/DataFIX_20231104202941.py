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
import tensorflow
import numba
import math
import sys
import time
# Path: Delivery/delivery_create.py

sys.path.append('Codes/alphavantage/')

import alphavant

#Find uncompleted columns in dataframe and fix them

fold = alphavant.StocksData(api_key='1INOZ30DO4QWK4KV')

def get_data(ticker):
    return fold.get_daily_adjusted_data(symbol=ticker)

print(get_data('TRP'))