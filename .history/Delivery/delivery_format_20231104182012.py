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

sys.path.append('Codes/Strat/')

import file_management 

# Create Delivery format or overwrite existing one

file_management=file_management.FileManagement()

submission_mock=file_management.load_data(folder_name='Data',file_name='sample_submission.csv')
column_name=submission_mock.columns

new_folder=pd.DataFrame(columns=column_name)
new_folder['date'] = [pd.NaT] * len(submission_mock)
new_folder['date'] = submission_mock['date']
print(new_folder)





