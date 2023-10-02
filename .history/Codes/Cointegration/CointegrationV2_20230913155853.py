import math
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import multiprocessing
from multiprocessing import Pool

import sys
sys.path.insert(0,'/Codes/WRDS')
sys.path.insert(0,'/Codes/Methods')


#########
# Change path to the WRDS folder
sys.path.insert(0,'D:\CDT-HFT\TradingClubHFT\Codes\WRDS')
#########



from Wrds_Connect import WRDSDataFetcher
from Wrds_Connect import NameCompiler


wrds_username   ="coffeebeans007"
wrds_password   ="Gauj7744Jcpower50$$"
library_name    ="taqm_2021"


connection=WRDSDataFetcher(library_name,wrds_username,wrds_password)
connection.connect()

dataset_name= NameCompiler.create_quert("complete_nbbo",2021,'Jun',4)
dataset=connection.get_dataset(dataset_name)
print(dataset)

connection.close()











