##### Function to add : Regression algorithm #####

from sklearn.linear_model import LinearRegression
import numpy as np


def linear_regression(x, y):
    """
    Perform a simple linear regression of y on x.
    
    Parameters:
    - x: Independent variable (array-like).
    - y: Dependent variable (array-like).
    
    Returns:
    - alpha (intercept), beta (slope), rsquared, std_err (standard error of beta).
    """
    A = np.vstack([x, np.ones(len(x))]).T
    beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Predicted values
    y_pred = alpha + beta * x
    
    # R squared calculation
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    rsquared = 1 - (ss_res / ss_tot)
    
    # Standard error of beta
    n = len(x)
    mse = np.sum((y_pred - y) ** 2) / (n - 2)
    var_beta = mse / np.sum((x - np.mean(x)) ** 2)
    std_err = np.sqrt(var_beta)
    
    return alpha, beta, rsquared, std_err


def more_optimized_regression(df_price, pairs_df, window_size=5):
    """
    Perform more optimized regression for given pairs using a sliding window approach.
    
    Parameters:
    - df_price: DataFrame with 'date', 'ticker', and 'price' columns.
    - pairs_df: DataFrame with pairs information.
    - window_size: Size of the sliding window (default is 5 days).
    
    Returns:
    - DataFrame with regression results.
    """
    results = []
    
    # Pivot the price dataframe
    df_pivoted = df_price.pivot(index='date', columns='ticker', values='price')
    
    for _, row in pairs_df.iterrows():
        pair_results = regression_for_pair_pivoted(df_pivoted, row['Stock 1'], row['Stock 2'], window_size)
        
        for res in pair_results:
            date, beta, alpha, rsquared, std_err = res
            results.append([date, row['Pair Name'], row['Cluster'], beta, alpha, rsquared, std_err])
    
    return pd.DataFrame(results, columns=['Date', 'Pair Name', 'Cluster', 'Beta', 'Alpha', 'R^2', 'Std(Beta)'])

# Running the more optimized regression process using the pivoted dataframe
df_more_optimized_regression = more_optimized_regression(df_price, df_pairs, window_size=2)
df_more_optimized_regression


# Redefining the regression function for the pivoted data
def regression_for_pair_pivoted(df_pivoted, stock_1, stock_2, window_size=5):
    """
    Perform regression for a given pair using a sliding window approach on a pivoted dataframe.
    
    Parameters:
    - df_pivoted: Pivoted DataFrame with dates as index and tickers as columns.
    - stock_1, stock_2: Names of the stocks in the pair.
    - window_size: Size of the sliding window (default is 5 days).
    
    Returns:
    - List of regression results for each date (alpha, beta, rsquared, std_err).
    """
    results = []
    dates = df_pivoted.index
    
    for idx, date in enumerate(dates):
        if idx < window_size - 1:
            continue
        
        # Extract data for the window
        stock_1_prices = df_pivoted[stock_1].iloc[idx - window_size + 1: idx + 1].dropna().to_numpy()
        stock_2_prices = df_pivoted[stock_2].iloc[idx - window_size + 1: idx + 1].dropna().to_numpy()
        
        if len(stock_1_prices) != window_size or len(stock_2_prices) != window_size:
            continue
        
        alpha, beta, rsquared, std_err = linear_regression(stock_1_prices, stock_2_prices)
        results.append([date, beta, alpha, rsquared, std_err])
    
    return results

# Running the more optimized regression process using the pivoted dataframe
df_more_optimized_regression = more_optimized_regression(df_price, df_pairs, window_size=2)
df_more_optimized_regression
