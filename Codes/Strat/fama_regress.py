import statsmodels.api as sm
import pandas as pd
from os_helper import OsHelper


class FamaFrenchRegression:
    """
    A class used to run regression analysis of portfolio returns against Fama-French factors.

    Attributes
    ----------
    ff_factors : pd.DataFrame
        DataFrame containing the Fama-French factors.
    portfolio_returns : pd.DataFrame
        DataFrame containing the portfolio returns.
    merged_data : pd.DataFrame
        DataFrame containing the merged data of portfolio returns and Fama-French factors.
    model : sm.OLS
        The Ordinary Least Squares regression model after fitting.
    
    Methods
    -------
    __init__(self, ff_factors_path: str, portfolio_returns_path: str)
        Initializes the class with paths to the Fama-French factors and portfolio returns CSV files.
    _preprocess_data(self)
        Preprocesses the data by renaming columns, converting to datetime, and merging datasets.
    run_regression(self)
        Runs the OLS regression using the Fama-French factors against the portfolio returns.
    get_regression_results(self) -> str
        Returns the summary of the regression results.
    """
    
    def __init__(self, ff_factors: pd.DataFrame, portfolio_returns: pd.DataFrame):
        """
        Constructs all the necessary attributes for the FamaFrenchRegression object.

        Parameters
        ----------
        ff_factors : pd.DataFrame
            DataFrame containing Fama-French factors data.
        portfolio_returns : pd.DataFrame
            DataFrame containing portfolio returns data.
        """
        # Assign data
        self.ff_factors = ff_factors
        self.portfolio_returns = portfolio_returns
        
        # Preprocess data
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Processes the loaded data to prepare it for regression analysis."""
        # Rename the 'Unnamed: 0' column to 'Date' and convert to datetime
        self.ff_factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        self.ff_factors['Date'] = pd.to_datetime(self.ff_factors['Date'], format='%Y%m%d')
        self.ff_factors.iloc[:, 1:] = self.ff_factors.iloc[:, 1:] / 100
        
        self.portfolio_returns.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        self.portfolio_returns['Date'] = pd.to_datetime(self.portfolio_returns['Date'])
        
        # Merge datasets on the 'Date' column
        self.merged_data = pd.merge(self.portfolio_returns, self.ff_factors, on='Date')
        
        # Subtract the Risk-Free rate (RF) from the portfolio returns to get excess returns
        self.merged_data['Excess_Return'] = self.merged_data['Portfolio_Returns'] - self.merged_data['RF']
        
        # Ensure the scales match (convert all to decimals if necessary)
        # Assuming all the scales are already in percentages and converted to decimals above
    
    def run_regression(self):
        """
        Runs the regression analysis using the Ordinary Least Squares (OLS) method.
        """
        # Define the independent variables (Fama-French factors) and the dependent variable (Excess_Return)
        X = self.merged_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        y = self.merged_data['Excess_Return']
        
        # Create a regression model and fit it
        self.model = sm.OLS(y, X).fit()
        
    def get_regression_results(self) -> str:
        """
        Retrieves the regression results summary.

        Returns
        -------
        str
            The summary of the regression results as a string.
        """
        return self.model.summary()
    def run_rolling_regression(self, window_size: int = 252) -> pd.DataFrame:
        """
        Runs rolling window regressions of the portfolio returns against the Fama-French factors.

        Parameters
        ----------
        window_size : int
            The number of trading days to include in each rolling window (default is 252, approximately one trading year).

        Returns
        -------
        pd.DataFrame
            A DataFrame with dates as the index and the regression coefficients for each factor as the columns.
        """
        # Define the independent variables (Fama-French factors)
        independent_vars = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        # Create an empty DataFrame to store the regression coefficients, with columns for each factor
        rolling_coeffs = pd.DataFrame(index=self.merged_data['Date'], columns=independent_vars)

        # Iterate over the merged data, shifting the window by one day each time
        for end_index in range(window_size, len(self.merged_data) + 1):
            # Define the start index for the rolling window
            start_index = end_index - window_size

            # Subset the data for the current window
            window_data = self.merged_data.iloc[start_index:end_index]
            X = window_data[independent_vars]
            y = window_data['Excess_Return']

            # Run the regression for the current window
            model = sm.OLS(y,X).fit()

            coeffs = model.params
            rolling_coeffs.loc[window_data.iloc[-1]['Date'], coeffs.index] = coeffs.values

        # Drop rows that have not been filled with coefficients due to the rolling window
        rolling_coeffs = rolling_coeffs.dropna()

        return rolling_coeffs
  
if __name__ == "__main__":
    # Paths to the CSV files containing the Fama-French factors and portfolio returns
    os_helper = OsHelper()
    ff_factors = os_helper.read_data(directory_name='benchmark', file_name='F-F_Research_Data_5_Factors_2x3_daily.csv')
    portfolio_returns = os_helper.read_data(directory_name='final data', file_name='portfolio_returns.csv')

    print(ff_factors.head())
    print(portfolio_returns.head())

    # Instantiate the FamaFrenchRegression class
    ff_regression = FamaFrenchRegression(ff_factors, portfolio_returns)

    # Run the regression
    ff_regression.run_regression()

    # Retrieve and display the regression results
    regression_results = ff_regression.get_regression_results()
    print(regression_results)
    
     # Run the rolling regression
    rolling_results = ff_regression.run_rolling_regression(window_size=252)

    # Print or use the rolling regression results
    print(rolling_results)

