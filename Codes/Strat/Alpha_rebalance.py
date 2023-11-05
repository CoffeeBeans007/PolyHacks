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
import file_management 


class Rolling_Rebalance:
    
    def __init__(self,windows,data) -> None:
        findfunction= file_management.FileManagement()
        self.windows = windows
        self.data = findfunction.load_data(data)
        pass
    
    
    def exponential_function(x, a):
        return 1 - math.exp(-a * x)


    @numba.jit
    def rolling_beta(self,windows,data,exponential=False,exponential_factor=0.5):
        """
        Cut datasets into windows and calculate beta for each window
        Inputs:
        - windows: int, number of windows
        - data: dataframe, dataframe of returns

        Outputs:
        - beta: dataframe, dataframe of betas
        """
        if exponential==False:
            """
            Cut-beta into n equally sized windows
            """
            beta = pd.DataFrame()
            for i in range(windows):
                data_window = data.iloc[0:(i)*int(len(data)/windows)]
                beta_window = data_window.apply(lambda x: np.cov(x,data['S&P'])[0][1]/np.var(data['S&P']),axis=0)
                beta = pd.concat([beta,beta_window],axis=1)
            beta.columns = [i for i in range(windows)]
            return beta
        elif exponential==True:
            """
            Cut-beta into n exponentially smaller windows using exponential factor
            """
            # Define the exponential function
            x=np.linspace(0,1,len(data))
            
            a=1
            y = [exponential_factor(xi, a) for xi in x]
            beta = pd.DataFrame()
            for i in range(windows):
                data_window = data.iloc[0:y[i]*len(data)]
                beta_window = data_window.apply(lambda x: np.cov(x,data['SPY'])[0][1]/np.var(data['SPY']),axis=0)
                beta = pd.concat([beta,beta_window],axis=1)
            beta.columns = [i for i in range(windows)]
            return beta
            
    def slow_rebalance(self,data,speed=0.5) -> pd.DataFrame:
        """
        Rebalance the portfolio slowly
        Inputs:
        - data: dataframe, dataframe of returns
        - speed: float, speed of rebalancing

        Outputs:
        - weightings: dataframe, dataframe of weightings
        """
        weightings = pd.DataFrame()
        for i in range(len(data)):
            if i==0:
                weightings = pd.concat([weightings,pd.DataFrame([1])],axis=1)
            else:
                weightings = pd.concat([weightings,pd.DataFrame([weightings.iloc[i-1][0]*(1+data.iloc[i][0]*speed)])],axis=1)
        return weightings

        


