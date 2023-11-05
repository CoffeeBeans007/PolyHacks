from os_helper import OsHelper
import pandas as pd

class Filter:
    def __init__(self, data: pd.DataFrame, exclude_metrics: list = ["average_turnover"]):
        self.data = data.dropna().copy()
        self.exclude_metrics = exclude_metrics
        self.risk_cols = self._identify_risk_cols()

    def _identify_risk_cols(self):
        exclude_cols = ['Date', 'Ticker'] + [col for col in self.data.columns if any(metric in col for metric in self.exclude_metrics)]
        risk_cols = [col for col in self.data.columns if col not in exclude_cols]
        return risk_cols

    def filter_liquidity(self, top_n: int, liquidity_metric: str = "average_turnover"):
        liquidity_cols = [col for col in self.data.columns if liquidity_metric in col]
        liquidity_data = self.data.copy()
        liquidity_data[liquidity_cols] = liquidity_data[liquidity_cols].apply(pd.to_numeric, errors='coerce')
        liquidity_data['Liquidity_Mean'] = liquidity_data[liquidity_cols].mean(axis=1)
        liquidity_data['Liquidity_Rank'] = liquidity_data.groupby('Date')['Liquidity_Mean'].rank()
        filtered_data = liquidity_data[liquidity_data['Liquidity_Rank'] <= top_n].copy()
        filtered_data.drop(columns=['Liquidity_Mean', 'Liquidity_Rank'], inplace=True)
        return filtered_data

    def filter_risk_metrics(self, data: pd.DataFrame, top_n: int):
        risk_data = data.copy()
        risk_data[self.risk_cols] = risk_data[self.risk_cols].apply(pd.to_numeric, errors='coerce')
        risk_data['Avg_Risk_Score'] = risk_data[self.risk_cols].mean(axis=1)
        risk_data['Risk_Rank'] = risk_data.groupby('Date')['Avg_Risk_Score'].rank()
        filtered_data = risk_data[risk_data['Risk_Rank'] <= top_n].copy()
        filtered_data.drop(columns=['Avg_Risk_Score', 'Risk_Rank'], inplace=True)
        return filtered_data

    def apply_filters(self, liquidity_top_n: int, risk_top_n: int):
        liquidity_filtered_data = self.filter_liquidity(top_n=liquidity_top_n)
        risk_filtered_data = self.filter_risk_metrics(data=liquidity_filtered_data, top_n=risk_top_n)
        risk_filtered_data.reset_index(drop=True, inplace=True)
        return risk_filtered_data


# Exemple d'utilisation
if __name__ == "__main__":
    os_helper = OsHelper()
    data = os_helper.read_data(directory_name="transform filtered_data", file_name="rebalance_metrics.csv", index_col=0)
    print(data)
    risk_filter = Filter(data=data)
    filtered_data = risk_filter.apply_filters(liquidity_top_n=200, risk_top_n=120)
    print(filtered_data.head())

    os_helper.write_data(directory_name="transform filtered_data", file_name="filtered_data.csv", data_frame=filtered_data)




