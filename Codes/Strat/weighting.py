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
        """
        Initializes the InverseMetricsWeighting object.

        Args:
            filtered_data (pd.DataFrame): The input data.
            years_list (list[float, int]): List of years for which metrics are computed.
            target_metric (str): The metric to compute.
            weight_list (list[float]): List of weights corresponding to the years.
            min_limit (float): Minimum limit for the weights.
            max_limit (float): Maximum limit for the weights.
        """
        self.filtered_data = self._ensure_data_format(data=filtered_data)
        self.years_list = self._confirm_years(years_list)
        self.target_metric = self._confirm_metric(target_metric)
        self.weight_list = self._confirm_weights(weight_list)
        self.min_limit, self.max_limit = self._ensure_weight_limits(min_limit=min_limit, max_limit=max_limit)

    def _confirm_metric(self, metric: str) -> str:
        """
        Validates the target metric against available metrics.

        Args:
            metric (str): The target metric to validate.

        Returns:
            str: The confirmed metric.

        Raises:
            ValueError: If the metric is not available.
        """
        available_metrics = self._fetch_unique_metrics()
        if metric not in available_metrics:
            raise ValueError(f"The metric '{metric}' is not in the list of available metrics: {available_metrics}")
        return metric

    def _confirm_years(self, years: List[Union[float, int]]) -> List[Union[float, int]]:
        """
        Confirm that the specified years are available in the dataset.

        Args:
            years (List[Union[float, int]]): A list of years to check.

        Returns:
            List[Union[float, int]]: The same list of years if all are valid.

        Raises:
            ValueError: If the list of years is empty or if a year is not available in the dataset.
        """
        if not years:
            raise ValueError("years must be a list of at least one year")

        available_years = self._fetch_unique_years()
        for year in years:
            if year not in available_years:
                raise ValueError(f"The year {year} is not in the list of available years: {available_years}")

        return years

    @staticmethod
    def _ensure_data_format(data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the date column is in datetime format and filters out unwanted metrics.

        Args:
            data (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The formatted dataframe.

        Raises:
            ValueError: If the date column cannot be converted to datetime.
        """
        if not isinstance(data['Date'], pd.DatetimeIndex):
            try:
                data['Date'] = pd.to_datetime(data['Date'])
                data.index.name = None
            except ValueError:
                raise ValueError()

        excluded_metrics = "average_turnover"
        return data.filter(regex=f"^(?!{excluded_metrics}).*$")

    def _fetch_unique_years(self) -> list[str]:
        """
        Fetch unique years from the column names of the filtered data.

        Returns:
            list[str]: A list of unique years.
        """

        columns = self.filtered_data.columns
        year_nums = [re.findall(r'\d+', col) for col in columns]
        year_nums = [int(num[0]) for num in year_nums if len(num) > 0]
        year_nums = list(set(year_nums))
        year_nums.sort()
        return year_nums

    def _fetch_unique_metrics(self) -> list[str]:
        """
        Fetch unique metrics from the column names of the filtered data.

        Returns:
            list[str]: A list of unique metrics.
        """
        columns = self.filtered_data.columns
        metrics = [re.findall(r'[a-zA-Z]+', col) for col in columns if col not in ['Date', 'Ticker']]
        metrics = [metric[0] for metric in metrics if len(metric) > 0]
        metrics = list(set(metrics))
        metrics.sort()
        return metrics

    def _confirm_weights(self, weights: list) -> list:
        """
        Validate and normalize the provided weights for pondering metrics. If metric to inverse is volatility and
        rolling years are 1 and 3, then weights could be [0.5, 0.5].

        Args:
            weights (list): A list of weights.

        Returns:
            list: The validated list of weights.

        Raises:
            ValueError: If the length of weights does not match the length of years or if the sum is not equal to 1.
        """
        if not weights:
            weights = [1.0 / len(self._fetch_unique_years())] * len(self._fetch_unique_years())

        if len(weights) != len(self._fetch_unique_years()):
            raise ValueError("The length of weights does not match the length of years.")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("The sum of weights is not equal to 1.")

        return weights

    @staticmethod
    def _ensure_weight_limits(min_limit: float, max_limit: float) -> Tuple[float, float]:
        """
        Validate the weight limits.

        Args:
            min_limit (float): The minimum limit.
            max_limit (float): The maximum limit.

        Returns:
            Tuple[float, float]: The validated weight limits.

        Raises:
            ValueError: If the limits are invalid.
        """
        if not isinstance(min_limit, float) or not isinstance(max_limit, float) or not 0 <= min_limit < max_limit <= 1:
            raise ValueError("Invalid weight constraints.")
        return min_limit, max_limit

    def _select_metric_columns(self) -> pd.DataFrame:
        """
        Selects columns related to the target metric for the specified years.

        Returns:
            pd.DataFrame: The dataframe with the selected columns.
        """
        metrics_columns = [f"{self.target_metric}_{year}Y" for year in self.years_list]
        selected_columns = ["Date", "Ticker"] + metrics_columns
        return self.filtered_data[selected_columns]

    def _compute_inverse_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the inverse of the selected metrics and return the updated dataframe.

        Args:
            metrics_df (pd.DataFrame): The input dataframe containing the metrics.

        Returns:
            pd.DataFrame: The updated dataframe with inverse metrics calculated.
        """
        # Iterate through years and corresponding weights
        for year, weight in zip(self.years_list, self.weight_list):
            # Construct the metric column name
            metric_column = f"{self.target_metric}_{year}Y"

            # Ensure beta values are non-negative
            if self.target_metric == "beta":
                metrics_df[metric_column] = metrics_df[metric_column].apply(lambda x: 0.001 if x < 0 else x)

            # Compute the inverse of the metric
            metrics_df[f"inverse_{metric_column}"] = 1 / metrics_df[metric_column]

        return metrics_df

    def _compute_weighted_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the weighted metrics and return the updated dataframe.

        Args:
            metrics_df (pd.DataFrame): The dataframe with inverse metrics calculated.

        Returns:
            pd.DataFrame: A dataframe containing the weighted metrics.
        """
        final_weights_df = pd.DataFrame()

        # Iterate through years and corresponding weights
        for year, weight in zip(self.years_list, self.weight_list):
            # Construct the metric column name
            metric_column = f"{self.target_metric}_{year}Y"

            # Compute the sum of inverse metrics grouped by date
            metric_sum = metrics_df.groupby("Date")[f"inverse_{metric_column}"].transform("sum")

            # Calculate and store the weighted metric
            final_weights_df[f"weighted_{metric_column}"] = weight * (
                        metrics_df[f"inverse_{metric_column}"] / metric_sum)

        return final_weights_df.sum(axis=1)

    def _optimize_weights(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize the weights using convex optimization and return the updated dataframe.

        Args:
            metrics_df (pd.DataFrame): The dataframe with weighted metrics.

        Returns:
            pd.DataFrame: The dataframe with optimized weights.
        """
        # Iterate through unique dates
        for date in metrics_df["Date"].unique():
            # Extract initial weights for the date
            initial_weights = metrics_df.loc[metrics_df["Date"] == date, "Weight"].values

            # Define the optimization variable
            final_weights = cp.Variable(len(initial_weights))

            # Define the objective function
            objective = cp.Minimize(cp.sum_squares(final_weights - initial_weights))

            # Define constraints
            constraints = [cp.sum(final_weights) == 1, final_weights >= self.min_limit, final_weights <= self.max_limit]

            # Add constraints to ensure tickers with initial weights of 0 stay at 0
            for i, weight in enumerate(initial_weights):
                if weight == 0:
                    constraints.append(final_weights[i] == 0)

            # Solve the optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            # Update the dataframe with optimized weights
            metrics_df.loc[metrics_df["Date"] == date, "Weight"] = final_weights.value

        return metrics_df

    def compute_inverse_weighted_metrics(self):
        """
        Compute the inverse weighted metrics and adjust the weights using convex optimization.

        Returns:
            pd.DataFrame: A dataframe containing the date, ticker, and computed weights.

        Raises:
            ValueError: If the sum of weights for a given date is not close to 1.
        """
        # Select and copy the relevant metric columns
        metrics_df = self._select_metric_columns().copy()

        # Compute the inverse of the metrics
        metrics_df = self._compute_inverse_metrics(metrics_df)

        # Compute the initial weighted metrics
        metrics_df["Weight"] = self._compute_weighted_metrics(metrics_df)

        # Optimize the weights
        metrics_df = self._optimize_weights(metrics_df)

        # Extract and verify the final weighting
        weighting_df = metrics_df[["Date", "Ticker", "Weight"]]
        self._verify_weights_sum(weighting_df)

        return weighting_df

    def _verify_weights_sum(self, weighting_df: pd.DataFrame):
        """
        Verify that the sum of weights is close to 1 for each date.

        Args:
            weighting_df (pd.DataFrame): The dataframe containing date and computed weights.

        Raises:
            ValueError: If the sum of weights on any date is not equal to 1.
        """
        weight_sums = weighting_df.groupby("Date")["Weight"].sum().reset_index()
        for _, row in weight_sums.iterrows():
            date, weight_sum = row["Date"], row["Weight"]
            if not np.isclose(weight_sum, 1.0):
                raise ValueError(f"The sum of weights on date {date} is not equal to 1. Sum is {weight_sum}.")
        print("All weights sum to 1 for each date.")



if __name__ == "__main__":
    os_helper = OsHelper()
    filtered_data = os_helper.read_data(directory_name="transform data", file_name="filtered_data.csv", index_col=0)
    print(filtered_data)

    years_to_inverse = [1, 3]
    metric_to_inverse = "beta"
    weighting_list = [0.5, 0.5]
    min_weight = 0.00001
    max_weight = 0.05

    inverse_metrics_weighting = InverseMetricsWeighting(filtered_data=filtered_data, years_list=years_to_inverse,
                                                        target_metric=metric_to_inverse, weight_list=weighting_list,
                                                        min_limit=min_weight, max_limit=max_weight)
    inverse_metrics_weighting_df = inverse_metrics_weighting.compute_inverse_weighted_metrics()
    print(inverse_metrics_weighting_df)

    os_helper.write_data(directory_name="transform data", file_name="inverse_metrics_weighting.csv", data_frame=inverse_metrics_weighting_df)