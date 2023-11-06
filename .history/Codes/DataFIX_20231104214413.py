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
import yfinance as yf
# Path: Delivery/delivery_create.py

print(yf.download("AAPL", start="2017-01-01", end="2017-04-30"))

data=pd.read_csv('Data/tot_ret.csv')
# for columns in data that have NaN, replace with yf.download of the stock column name start is the earliest date in data[Date] and end is the latest date in data[Date]



for column_name in data.columns:
    if data[column_name].isna().any():
        # Get the earliest and latest date in data[Date]
        start_date = data['Date'].min()
        end_date = data['Date'].max()
        
        # Download the missing data using yfinance
        downloaded_data = yf.download(column_name, start=start_date, end=end_date)
        
        # Extract the 'Adj Close' column from the downloaded data
        downloaded_data = downloaded_data['Adj Close']
        
        # Replace NaN values in the original DataFrame with downloaded data
        data[column_name] = data[column_name].fillna(downloaded_data)