import statsmodels.api as sm
import pandas as pd

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
    
    def __init__(self, ff_factors_path: str, portfolio_returns_path: str):
        """
        Constructs all the necessary attributes for the FamaFrenchRegression object.

        Parameters
        ----------
        ff_factors_path : str
            The file path to the CSV file containing Fama-French factors data.
        portfolio_returns_path : str
            The file path to the CSV file containing portfolio returns data.
        """
        # Load data
        self.ff_factors = pd.read_csv(ff_factors_path)
        self.portfolio_returns = pd.read_csv(portfolio_returns_path)
        
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
  
if __name__ == "__main__":
    # Paths to the CSV files containing the Fama-French factors and portfolio returns
    ff_factors_path = '/Users/ced/Documents/PolyHacks/Data/benchmark/F-F_Research_Data_5_Factors_2x3_daily.CSV'
    portfolio_returns_path = '/Users/ced/Documents/PolyHacks/Data/transform data/portfolio_returns.csv'

    # Instantiate the FamaFrenchRegression class
    ff_regression = FamaFrenchRegression(ff_factors_path, portfolio_returns_path)

    # Run the regression
    ff_regression.run_regression()

    # Retrieve and display the regression results
    regression_results = ff_regression.get_regression_results()
    print(regression_results)
