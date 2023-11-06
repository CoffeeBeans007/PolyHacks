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

#Open csv
# Read the data and format DataFrames
data = pd.read_csv("Data/transform data/inverse_metrics_weighting.csv", sep=",")
format = pd.read_csv("Delivery/submission.csv")
format.set_index('date', inplace=True)
print(data)
# Merge the DataFrames on the 'date' and 'Date' columns
data.set_index('Date', inplace=True)

data.rename(columns={'Date': 'New_Column_Name'}, inplace=True)
data.drop(columns=['Unnamed: 0'], inplace=True)
# Pivot the DataFrame
pivoted_df=pd.read_csv("Data/transform data/drifted_weights.csv", sep=",")
pivoted_df.set_index('Unnamed: 0', inplace=True)
print(pivoted_df)
#remove "US Equity from all column names"
pivoted_df.columns = pivoted_df.columns.str.replace(' US Equity', '')
pivoted_df.columns = pivoted_df.columns.str.replace('_y', '')
pivoted_df.columns = pivoted_df.columns.str.replace('_x', '')
#Add "weight_" to all column names
pivot_df = pivoted_df.add_prefix('weight_')
pivot_df = pivot_df[pivot_df.index <= '2014-12-31']
print(pivot_df)

#Match columns and index between pivot_df and format
merged_df = format.combine_first(pivot_df)
merged_df["Rebal"]=np.where(merged_df.filter(like='weight_').notna().any(axis=1), 1, 0)
merged_df.columns = merged_df.columns.str.replace('_x', '')
merged_df.columns = merged_df.columns.str.replace('_y', '')

#switch the index and the id
merged_df.reset_index(inplace=True)
#Rename index as date
merged_df.rename(columns={'index': 'date'}, inplace=True)
#Remove id column

#Cut every row with date after 2014-12-31

merged_df = merged_df[merged_df['date'] <= '2014-12-31']


#set all nan to 0
merged_df.fillna(0, inplace=True)

column_name=merged_df.columns
# for j in column_name:
#     column_name=column_name.str.replace('weight_', '')
#     stock=j.replace('weight_', '')
   
#     if j!="date" and j!="id":
#     #If not only zero, outputs yf.download
#         minimumdatewithvalue=merged_df[j].ne(0).idxmax()

#         start=merged_df["date"].iloc[minimumdatewithvalue]
#         end=merged_df["date"].iloc[-1]
#         start=str(start)
#         end=str(end)
#         print(start,end)
#         print(column_name)
        
#         data=yf.download(str(stock), start=start, end=end)
#         if start not in data.index:
#             print("SJSJDASJDSAJDSADSAKJDKJSADSADKJSADK")
#             print(j)
#             merged_df[j]=0

            
merged_df.drop(columns=['Rebal'], inplace=True)      

merged_df.to_csv('Delivery/submissionss.csv', index=False)
#In merged_df

