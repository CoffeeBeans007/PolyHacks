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
from datetime import datetime
# Path: Delivery/delivery_create.py

data=pd.read_csv('Data/tot_ret.csv',sep=';')
# for columns in data that have NaN, replace with yf.download of the stock column name start is the earliest date in data[Date] and end is the latest date in data[Date]
print(data)


for column_name in data.columns:
    
    if data[column_name].isna().any():
        yf_name = column_name.replace(' US Equity', '')
        
        # Get the earliest and latest date in data['Date']
        start_date='1995-01-01'
        end_date='2015-12-31'
        
        # Download the missing data using yfinance
        downloaded_data = yf.download(yf_name, start=start_date, end=end_date)
        
        # Extract the 'Adj Close' column from the downloaded data
        downloaded_data = downloaded_data['Adj Close']


        
        data[column_name] = downloaded_data
    
        
        # Replace NaN values in the original DataFrame from the first NaN row onward
       
print(data)
# data=pd.read_csv('Data/base data/tot_ret.csv',sep=';')
# # for columns in data that have NaN, replace with yf.download of the stock column name start is the earliest date in data[Date] and end is the latest date in data[Date]

# data_marginal=data

# #Calculate the percentage of data returns (log_returns)

# for column in data.columns:
#     if column != 'Date':
#         data_marginal[column] = data[column].apply(lambda x: 0 if pd.isna(x) else x)
#         data_marginal[column] = data_marginal[column].pct_change().apply(lambda x: 0 if pd.isna(x) else x).apply(lambda x: 0 if x == -1 else x)*100
# print(data)

# data_marginal.to_csv('Data/base data/logged_total_returns.csv',sep=';',index=False)