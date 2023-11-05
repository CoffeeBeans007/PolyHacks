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

print(file_management.load_data(folder_name='Data',file_name='sector.csv'))





