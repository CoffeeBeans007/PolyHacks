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

class DataDelivery:
    def __init__(self, folder_name, file_name):
        self.folder_name = folder_name
        self.file_name = file_name

    def create_delivery(self):
        # Create Delivery format or overwrite existing one
        file_management_instance = file_management.FileManagement()
        submission_mock = file_management_instance.load_data(folder_name=self.folder_name, file_name=self.file_name)
        column_name = submission_mock.columns

        new_folder = pd.DataFrame(columns=column_name)
        new_folder['date'] = [pd.NaT] * len(submission_mock)
        new_folder['date'] = submission_mock['date']
        new_folder['id'] = submission_mock['id']

        new_folder.to_csv('Delivery/submissionss.csv', index=False)





