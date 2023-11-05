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

import delivery_format
sys.path.append('Codes/Strat/')

import file_management 

file_management_instance = file_management.FileManagement()


# Create a delivery format csv then load it
format=delivery_format.DataDelivery('Data','sample_submission.csv')
create=format.create_delivery

submission_df = file_management_instance.load_data(folder_name='Delivery', file_name='submission.csv')


