from os_helper import OsHelper
import pandas as pd

class Filter:
    def __init__(self, data: pd.DataFrame, exclude_metrics: list = ["average_turnover"]):
        """
        Initializes the Filter class.

        Args:
            data (pd.DataFrame): The input data.
            exclude_metrics (list): List of metrics to exclude from risk columns.
        """
        self.data = data.dropna().copy()
        self.exclude_metrics = exclude_metrics
        self.risk_cols = self._identify_risk_cols()

    def _identify_risk_cols(self):
        """
        Identifies risk columns based on exclude metrics.

        Returns:
            list: List of risk columns.
        """
        # Exclude columns that are not to be considered as risk metrics
        exclude_cols = ['Date', 'Ticker'] + [col for col in self.data.columns if any(metric in col for metric in self.exclude_metrics)]
        # Identify risk columns
        risk_cols = [col for col in self.data.columns if col not in exclude_cols]
        return risk_cols

    def filter_liquidity(self, top_n: int, liquidity_metric: str = "average_turnover"):
        """
        Filters the data based on liquidity.

        Args:
            top_n (int): Top N rows to consider based on liquidity ranking.
            liquidity_metric (str): The metric used for liquidity ranking.

        Returns:
            pd.DataFrame: Data filtered by liquidity.
        """
        # Identify columns related to liquidity
        liquidity_cols = [col for col in self.data.columns if liquidity_metric in col]
        liquidity_data = self.data.copy()
        # Ensure numeric conversion
        liquidity_data[liquidity_cols] = liquidity_data[liquidity_cols].apply(pd.to_numeric, errors='coerce')
        # Calculate mean liquidity and rank by date
        liquidity_data['Liquidity_Mean'] = liquidity_data[liquidity_cols].mean(axis=1)
        liquidity_data['Liquidity_Rank'] = liquidity_data.groupby('Date')['Liquidity_Mean'].rank()
        # Filter data based on liquidity rank
        filtered_data = liquidity_data[liquidity_data['Liquidity_Rank'] <= top_n].copy()
        filtered_data.drop(columns=['Liquidity_Mean', 'Liquidity_Rank'], inplace=True)
        return filtered_data

    def filter_risk_metrics(self, data: pd.DataFrame, top_n: int):
        """
        Filters the data based on risk metrics.

        Args:
            data (pd.DataFrame): The input data.
            top_n (int): Top N rows to consider based on risk ranking.

        Returns:
            pd.DataFrame: Data filtered by risk metrics.
        """
        risk_data = data.copy()
        # Ensure numeric conversion
        risk_data[self.risk_cols] = risk_data[self.risk_cols].apply(pd.to_numeric, errors='coerce')
        # Calculate average risk score and rank by date
        risk_data['Avg_Risk_Score'] = risk_data[self.risk_cols].mean(axis=1)
        risk_data['Risk_Rank'] = risk_data.groupby('Date')['Avg_Risk_Score'].rank()
        # Filter data based on risk rank
        filtered_data = risk_data[risk_data['Risk_Rank'] <= top_n].copy()
        filtered_data.drop(columns=['Avg_Risk_Score', 'Risk_Rank'], inplace=True)
        return filtered_data

    def apply_filters(self, liquidity_top_n: int, risk_top_n: int):
        """
        Applies both liquidity and risk filters to the data.

        Args:
            liquidity_top_n (int): Top N rows to consider based on liquidity ranking.
            risk_top_n (int): Top N rows to consider based on risk ranking.

        Returns:
            pd.DataFrame: Data filtered by both liquidity and risk metrics.
        """
        # Apply liquidity filter
        liquidity_filtered_data = self.filter_liquidity(top_n=liquidity_top_n)
        # Apply risk metrics filter on liquidity filtered data
        risk_filtered_data = self.filter_risk_metrics(data=liquidity_filtered_data, top_n=risk_top_n)
        risk_filtered_data.reset_index(drop=True, inplace=True)
        return risk_filtered_data


# Exemple d'utilisation
if __name__ == "__main__":
    os_helper = OsHelper()
    data = os_helper.read_data(directory_name="transform data", file_name="rebalance_metrics.csv", index_col=0)
    print(data)
    risk_filter = Filter(data=data)

    risk_top_n = 120
    liquidity_top_n = 200

    filtered_data = risk_filter.apply_filters(liquidity_top_n=liquidity_top_n, risk_top_n=risk_top_n)
    print(filtered_data.head())

    os_helper.write_data(directory_name="transform data", file_name="filtered_data.csv", data_frame=filtered_data)




