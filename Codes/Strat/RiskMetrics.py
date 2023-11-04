import pandas as pd

class RiskMetrics:
    def __init__(self,benchmark_directory, returns_directory) -> None:
        self.benchmark = pd.read_csv(benchmark_directory)
        self.benchmark['benchmark_return'] = self.benchmark['benchmark_return'] - 1
        self.returns = pd.read_csv(returns_directory)
        
if __name__ == "__main__":
    test = RiskMetrics("Data/benchmark/SPY_returns.csv")
    print(test.benchmark)