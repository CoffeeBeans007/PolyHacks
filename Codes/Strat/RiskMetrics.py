import pandas as pd
import datetime
import os
import numpy as np
from typing import Union, Set, Any
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

class RiskMetrics(object):
    def __init__(self, daily_data: pd.DataFrame, benchmark_name: str, frequency: str = "D", years_metrics_list: list = None,
                 weekday_resampling: str = "FRI"):

        self.daily_data = daily_data
        self.benchmark_name = benchmark_name
        self.frequency = frequency
        self.years_metrics_list = years_metrics_list
        self.weekday_resampling = weekday_resampling



if __name__ == '__main__':
    pass
