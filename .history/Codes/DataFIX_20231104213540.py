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

df=pd.read_csv('Data/tot_ret.csv',sep=';')

# Define your Alpha Vantage API key
api_key = '1INOZ30DO4QWK4KV'


# Convert the 'Date' column to datetime, if it's not already
df['Date'] = pd.to_datetime(df['Date'])

# Create a function to fetch historical data from Alpha Vantage
def fetch_historical_data(ticker_symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker_symbol}&outputsize=full&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    if 'Time Series (Daily)' in data:
        daily_data = data['Time Series (Daily)']
        print(daily_data)

            # return daily_data[date]['4. close'], daily_data[date]['7. dividend amount']
    
    return None, None

fetch_historical_data('AAPL')
