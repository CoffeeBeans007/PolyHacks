import quantstats as qs
import pandas as pd
from typing import Dict, Union
from os_helper import OsHelper

class PerfReport(object):
    def __init__(self, port_returns: pd.DataFrame, bench_returns: Union[pd.DataFrame, None], strat_name: str = "InverseMetrics"):
        self.port_returns = self._prep_port_returns(port_returns=port_returns)
        self.bench_returns = self._prep_bench_returns(bench_returns=bench_returns)
        self.strat_name = strat_name

    @staticmethod
    def _prep_port_returns(port_returns: pd.DataFrame) -> pd.DataFrame:
        if port_returns.index.dtype != "datetime64[ns]":
            port_returns.index = pd.to_datetime(port_returns.index, format="%Y-%m-%d")
        if "Portfolio_Returns" not in port_returns.columns:
            raise ValueError()
        if port_returns["Portfolio_Returns"].isnull().any():
            raise ValueError()
        return port_returns

    @staticmethod
    def _prep_bench_returns(bench_returns: pd.DataFrame) -> Union[pd.DataFrame, None]:
        if bench_returns is None:
            return bench_returns
        if bench_returns.index.dtype != "datetime64[ns]":
            bench_returns.index = pd.to_datetime(bench_returns.index, format="%Y-%m-%d")
        bench_returns = bench_returns.pct_change().dropna()
        return bench_returns

    @staticmethod
    def _transfer_file(dest_folder: str, file_name: str) -> None:
        os_helper = OsHelper()
        os_helper.move_file(src_file_name=file_name, dest_directory_name=dest_folder)

    def generate_html_report(self, rf: float, periods_per_year: int, grayscale: bool = False, output: bool = True, match_dates: bool = True) -> None:
        qs.reports.html(returns=self.port_returns["Portfolio_Returns"],
                        benchmark=self.bench_returns,
                        rf=rf,
                        periods_per_year=periods_per_year,
                        title=f"{self.strat_name} strategy",
                        output=output,
                        grayscale=grayscale,
                        download_filename=f"{self.strat_name}_backtesting_report.html",
                        match_dates=match_dates)
        self._transfer_file(dest_folder=f"reports", file_name=f"{self.strat_name}_backtesting_report.html")

    def generate_full_report(self, rf: float, grayscale: bool = False, match_dates: bool = True) -> None:
        qs.reports.full(self.port_returns["Portfolio_Returns"], self.bench_returns, rf, grayscale, match_dates)

    def compute_metrics(self, rf: float, mode: str = "full", prepare_returns: bool = False, match_dates: bool = True) -> None:
        qs.reports.metrics(self.port_returns["Portfolio_Returns"], self.bench_returns, rf, mode, prepare_returns, match_dates)

    def get_performance_indicators(self, rf: float, periods_per_year: int, annualize: bool = True) -> Dict[str, float]:
        port_returns = self.port_returns["Portfolio_Returns"]
        cumulative_return = qs.stats.comp(port_returns)
        cagr = qs.stats.cagr(port_returns, rf=rf, compounded=True)
        sharpe = qs.stats.sharpe(port_returns, rf=rf, periods=periods_per_year, annualize=annualize)
        volatility = qs.stats.volatility(port_returns, periods=periods_per_year, annualize=annualize)
        max_drawdown = qs.stats.max_drawdown(port_returns)
        return {"Cumulative Returns": cumulative_return, "CAGR": cagr, "Sharpe": sharpe, "Volatility": volatility, "Max Drawdown": max_drawdown}


if __name__ == "__main__":
    os_helper = OsHelper()
    benchmark_prices = os_helper.read_data(directory_name="benchmark", file_name="SPY.csv", index_col=0)
    benchmark_prices.set_index(benchmark_prices.columns[0], inplace=True)
    benchmark_prices.sort_index(inplace=True)
    benchmark_prices.index.name = None
    benchmark_prices = benchmark_prices[["adjusted_close"]]
    benchmark_prices.columns = ["SPY"]

    print(benchmark_prices.head())

    strategy_returns = os_helper.read_data(directory_name="final data", file_name="portfolio_returns.csv", index_col=0)
    print(strategy_returns.head())
    strategy_returns.columns = ["Portfolio_Returns"]

    perf_report = PerfReport(port_returns=strategy_returns, bench_returns=benchmark_prices, strat_name="InverseMetrics")
    print(perf_report.get_performance_indicators(rf=0.01, periods_per_year=252))
    perf_report.generate_html_report(rf=0.01, periods_per_year=252)
