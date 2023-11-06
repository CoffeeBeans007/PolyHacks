import quantstats as qs
import pandas as pd
from typing import Dict, Union
from os_helper import OsHelper

import quantstats as qs
import pandas as pd
from typing import Dict, Union
from os_helper import OsHelper

class PerfReport(object):
    """
    Class to analyze and generate performance reports of a trading strategy.

    Attributes:
        port_returns (pd.DataFrame): Portfolio returns.
        bench_returns (Union[pd.DataFrame, None]): Benchmark returns.
        strat_name (str): Strategy name.
    """

    def __init__(self, port_returns: pd.DataFrame, bench_returns: Union[pd.DataFrame, None], strat_name: str = "InverseMetrics"):
        """
        Initializes the PerfReport class with portfolio and benchmark returns.

        Args:
            port_returns (pd.DataFrame): DataFrame containing portfolio returns.
            bench_returns (Union[pd.DataFrame, None]): DataFrame containing benchmark returns or None.
            strat_name (str, optional): Name of the strategy. Defaults to "InverseMetrics".
        """
        self.port_returns = self._prep_port_returns(port_returns=port_returns)
        self.bench_returns = self._prep_bench_returns(bench_returns=bench_returns)
        self.strat_name = strat_name

    @staticmethod
    def _prep_port_returns(port_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares portfolio returns for analysis.

        Args:
            port_returns (pd.DataFrame): Raw portfolio returns.

        Returns:
            pd.DataFrame: Processed portfolio returns.
        """
        # Ensure index is in datetime format
        if port_returns.index.dtype != "datetime64[ns]":
            port_returns.index = pd.to_datetime(port_returns.index, format="%Y-%m-%d")
        # Validate necessary columns
        if "Portfolio_Returns" not in port_returns.columns:
            raise ValueError("Expected column 'Portfolio_Returns' not found.")
        # Check for missing values
        if port_returns["Portfolio_Returns"].isnull().any():
            raise ValueError("Portfolio returns contain null values.")
        return port_returns

    @staticmethod
    def _prep_bench_returns(bench_returns: pd.DataFrame) -> Union[pd.DataFrame, None]:
        """
        Prepares benchmark returns for analysis.

        Args:
            bench_returns (pd.DataFrame): Raw benchmark returns.

        Returns:
            Union[pd.DataFrame, None]: Processed benchmark returns or None.
        """
        # Return None if no benchmark returns provided
        if bench_returns is None:
            return bench_returns
        # Ensure index is in datetime format
        if bench_returns.index.dtype != "datetime64[ns]":
            bench_returns.index = pd.to_datetime(bench_returns.index, format="%Y-%m-%d")
        # Convert to percentage change
        bench_returns = bench_returns.pct_change().dropna()
        return bench_returns

    @staticmethod
    def _transfer_file(dest_folder: str, file_name: str) -> None:
        """
        Transfers a file to the specified destination folder.

        Args:
            dest_folder (str): Destination folder.
            file_name (str): Name of the file to transfer.
        """
        os_helper = OsHelper()
        os_helper.move_file(src_file_name=file_name, dest_directory_name=dest_folder)

    def generate_html_report(self, rf: float, periods_per_year: int, grayscale: bool = False, output: bool = True, match_dates: bool = True) -> None:
        """
        Generates an HTML report of the strategy performance.

        Args:
            rf (float): Risk-free rate.
            periods_per_year (int): Number of periods per year.
            grayscale (bool, optional): Generate report in grayscale. Defaults to False.
            output (bool, optional): Display the report inline. Defaults to True.
            match_dates (bool, optional): Align dates of portfolio and benchmark. Defaults to True.
        """
        # Generate HTML report
        qs.reports.html(returns=self.port_returns["Portfolio_Returns"],
                        benchmark=self.bench_returns,
                        rf=rf,
                        periods_per_year=periods_per_year,
                        title=f"{self.strat_name} strategy",
                        output=output,
                        grayscale=grayscale,
                        download_filename=f"{self.strat_name}_backtesting_report.html",
                        match_dates=match_dates)
        # Transfer the report to the specified folder
        self._transfer_file(dest_folder=f"reports", file_name=f"{self.strat_name}_backtesting_report.html")

    def get_performance_indicators(self, rf: float, periods_per_year: int, annualize: bool = True) -> Dict[str, float]:
        """
        Computes and returns several performance indicators for the strategy.

        Args:
            rf (float): Risk-free rate.
            periods_per_year (int): Number of periods per year.
            annualize (bool): Flag to annualize metrics. Defaults to True.

        Returns:
            Dict[str, float]: Dictionary containing various performance indicators.
        """
        port_returns = self.port_returns["Portfolio_Returns"]
        cumulative_return = qs.stats.comp(port_returns)
        cagr = qs.stats.cagr(port_returns, rf=rf, compounded=True)
        sharpe = qs.stats.sharpe(port_returns, rf=rf, periods=periods_per_year, annualize=annualize)
        sortino = qs.stats.sortino(port_returns, rf=rf, periods=periods_per_year, annualize=annualize)
        calmar = qs.stats.calmar(port_returns)
        max_drawdown = qs.stats.max_drawdown(port_returns)
        avg_return = qs.stats.avg_return(port_returns)
        volatility = qs.stats.volatility(port_returns, periods=periods_per_year, annualize=annualize)
        tail_ratio = qs.stats.tail_ratio(port_returns)
        gain_to_pain_ratio = qs.stats.gain_to_pain_ratio(port_returns)
        value_at_risk = qs.stats.value_at_risk(port_returns)
        conditional_value_at_risk = qs.stats.conditional_value_at_risk(port_returns)
        kelly_criterion = qs.stats.kelly_criterion(port_returns)
        risk_of_ruin = qs.stats.risk_of_ruin(port_returns)
        win_rate = qs.stats.win_rate(port_returns)

        return {
            "Cumulative Returns": cumulative_return,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "Max Drawdown": max_drawdown,
            "Average Return": avg_return,
            "Volatility": volatility,
            "Tail Ratio": tail_ratio,
            "Gain to Pain Ratio": gain_to_pain_ratio,
            "Value at Risk (VaR)": value_at_risk,
            "Conditional Value at Risk (CVaR)": conditional_value_at_risk,
            "Kelly Criterion": kelly_criterion,
            "Risk of Ruin": risk_of_ruin,
            "Win Rate": win_rate,
        }


if __name__ == "__main__":
    os_helper = OsHelper()
    benchmark_prices = os_helper.read_data(directory_name="benchmark", file_name="SPY (1).csv", index_col=0)
    benchmark_prices.sort_index(inplace=True)
    benchmark_prices.index.name = None
    benchmark_prices = benchmark_prices[["Adj Close"]]
    benchmark_prices.columns = ["SPY"]

    print(benchmark_prices.head())

    strategy_returns = os_helper.read_data(directory_name="final data", file_name="portfolio_returns.csv", index_col=0)
    print(strategy_returns.head())
    strategy_returns.columns = ["Portfolio_Returns"]

    perf_report = PerfReport(port_returns=strategy_returns, bench_returns=benchmark_prices, strat_name="InverseMetrics")
    print(perf_report.get_performance_indicators(rf=0.01, periods_per_year=252))
    perf_report.generate_html_report(rf=0.01, periods_per_year=252)
