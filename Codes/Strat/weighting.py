import numpy as np
import pandas as pd
import re
from typing import Tuple, List, Union
import cvxpy as cp
from os_helper import OsHelper

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)


class InverseMetricsWeighting(object):
    def __init__(self, filtered_data: pd.DataFrame, years_list: list[float, int], target_metric: str,
                 weight_list: list[float], min_limit: float, max_limit: float):
        self.filtered_data = self._ensure_data_format(data=filtered_data)
        self.years_list = self._confirm_years(years_list)
        self.target_metric = self._confirm_metric(target_metric)
        self.weight_list = self._confirm_weights(weight_list)
        self.min_limit, self.max_limit = self._ensure_weight_limits(min_limit=min_limit, max_limit=max_limit)

    def _confirm_metric(self, metric: str):
        available_metrics = self._fetch_unique_metrics()
        if metric not in available_metrics:
            raise ValueError(f"The metric '{metric}' is not in the list of available metrics: {available_metrics}")
        return metric

    def _confirm_years(self, years: List[Union[float, int]]) -> List[Union[float, int]]:
        if not years:
            raise ValueError("years must be a list of at least one year")

        available_years = self._fetch_unique_years()
        for year in years:
            if year not in available_years:
                raise ValueError(f"The year {year} is not in the list of available years: {available_years}")

        return years

    @staticmethod
    def _ensure_data_format(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data['Date'], pd.DatetimeIndex):
            try:
                data['Date'] = pd.to_datetime(data['Date'])
                data.index.name = None
            except ValueError:
                raise ValueError()

        excluded_metrics = "average_turnover"
        return data.filter(regex=f"^(?!{excluded_metrics}).*$")

    def _fetch_unique_years(self) -> list[str]:
        columns = self.filtered_data.columns
        year_nums = [re.findall(r'\d+', col) for col in columns]
        year_nums = [int(num[0]) for num in year_nums if len(num) > 0]
        year_nums = list(set(year_nums))
        year_nums.sort()
        return year_nums

    def _fetch_unique_metrics(self) -> list[str]:
        columns = self.filtered_data.columns
        metrics = [re.findall(r'[a-zA-Z]+', col) for col in columns if col not in ['Date', 'Ticker']]
        metrics = [metric[0] for metric in metrics if len(metric) > 0]
        metrics = list(set(metrics))
        metrics.sort()
        return metrics

    def _confirm_weights(self, weights: list) -> list:
        if not weights:
            weights = [1.0 / len(self._fetch_unique_years())] * len(self._fetch_unique_years())

        if len(weights) != len(self._fetch_unique_years()):
            raise ValueError("The length of weights does not match the length of years.")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("The sum of weights is not equal to 1.")

        return weights

    @staticmethod
    def _ensure_weight_limits(min_limit: float, max_limit: float) -> Tuple[float, float]:
        if not isinstance(min_limit, float) or not isinstance(max_limit, float) or not 0 <= min_limit < max_limit <= 1:
            raise ValueError("Invalid weight constraints.")
        return min_limit, max_limit

    def _select_metric_columns(self) -> pd.DataFrame:
        metrics_columns = [f"{self.target_metric}_{year}Y" for year in self.years_list]
        selected_columns = ["Date", "Ticker"] + metrics_columns
        return self.filtered_data[selected_columns]

    def compute_inverse_weighted_metrics(self):
        metrics_df = self._select_metric_columns().copy()
        final_weights_df = pd.DataFrame()

        for year, weight in zip(self.years_list, self.weight_list):
            metric_column = f"{self.target_metric}_{year}Y"
            if self.target_metric == "beta":
                metrics_df[metric_column] = metrics_df[metric_column].apply(lambda x: 0.0001 if x < 0 else x)
            metrics_df[f"inverse_{metric_column}"] = 1 / metrics_df[metric_column]
            metric_sum = metrics_df.groupby("Date")[f"inverse_{metric_column}"].transform("sum")
            final_weights_df[f"weighted_{metric_column}"] = weight * (metrics_df[f"inverse_{metric_column}"] / metric_sum)

        metrics_df["Weight"] = final_weights_df.sum(axis=1)

        for date in metrics_df["Date"].unique():
            initial_weights = metrics_df.loc[metrics_df["Date"] == date, "Weight"].values
            zero_weights_indices = initial_weights == 0
            final_weights = cp.Variable(len(initial_weights))
            objective = cp.Minimize(cp.sum_squares(final_weights - initial_weights))
            constraints = [cp.sum(final_weights) == 1, final_weights >= self.min_limit, final_weights <= self.max_limit]
            for i in range(len(initial_weights)):
                if zero_weights_indices[i]:
                    constraints.append(final_weights[i] == 0)
            problem = cp.Problem(objective, constraints)
            problem.solve()
            metrics_df.loc[metrics_df["Date"] == date, "Weight"] = final_weights.value

        weighting_df = metrics_df[["Date", "Ticker", "Weight"]]
        return weighting_df



if __name__ == "__main__":
    os_helper = OsHelper()
    filtered_data = os_helper.read_data(directory_name="transform data", file_name="filtered_data.csv", index_col=0)
    print(filtered_data)

    years_to_inverse = [1, 3]
    metric_to_inverse = "beta"
    weighting_list = [0.5, 0.5]
    min_weight = 0.0
    max_weight = 0.05

    inverse_metrics_weighting = InverseMetricsWeighting(filtered_data=filtered_data, years_list=years_to_inverse,
                                                        target_metric=metric_to_inverse, weight_list=weighting_list,
                                                        min_limit=min_weight, max_limit=max_weight)
    inverse_metrics_weighting_df = inverse_metrics_weighting.compute_inverse_weighted_metrics()
    print(inverse_metrics_weighting_df)


    # os_helper.write_data(directory_name="transform filtered_data", file_name="inverse_metrics_weighting.csv", data_frame=inverse_metrics_weighting)