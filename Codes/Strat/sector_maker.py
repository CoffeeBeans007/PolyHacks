import pandas as pd

# Load the data
sectors_df = pd.read_csv('/Users/ced/Documents/PolyHacks/Data/base data/sectors.csv')
tot_retl_df = pd.read_csv('/Users/ced/Documents/PolyHacks/Data/base data/tot_retl.csv')

# Convert 'Date' to datetime in the 'tot_retl.csv' DataFrame
tot_retl_df['Date'] = pd.to_datetime(tot_retl_df['Date'])

# Calculate the daily returns for each stock
daily_returns_stocks_df = tot_retl_df.set_index('Date').pct_change().dropna() * 100

# Create a mapping of stocks to sectors
sectors_mapping = sectors_df.iloc[0].to_dict()

# Calculate the daily average return for each sector
daily_sector_avg_df = daily_returns_stocks_df.rename(columns=sectors_mapping).groupby(level=0, axis=1).mean()

# Calculate weekly returns for each stock
weekly_returns_stocks_df = daily_returns_stocks_df.resample('W').agg(['mean', 'min', 'std'])

# Flatten the MultiIndex created by resample
weekly_returns_stocks_df.columns = [' '.join(col).strip() for col in weekly_returns_stocks_df.columns.values]

# Weekly average returns by sector
weekly_avg_returns_by_sector = weekly_returns_stocks_df.filter(regex='mean$').rename(columns=sectors_mapping).groupby(level=0, axis=1).mean()

# Weekly maximum drawdown by sector
weekly_drawdown_by_sector = weekly_returns_stocks_df.filter(regex='min$').rename(columns=sectors_mapping).groupby(level=0, axis=1).min()

# Weekly volatility by sector
volatility_mapping = {f"{stock} std": sector for stock, sector in sectors_mapping.items()}
weekly_volatility_by_sector = weekly_returns_stocks_df.filter(regex='std$').rename(columns=volatility_mapping).groupby(level=0, axis=1).mean()

# Save results to CSV
daily_sector_avg_df.to_csv('/Users/ced/Documents/PolyHacks/Data/sector_Data/daily_sector_avg_returns.csv')
weekly_avg_returns_by_sector.to_csv('/Users/ced/Documents/PolyHacks/Data/sector_Data/weekly_avg_returns_by_sector.csv')
weekly_drawdown_by_sector.to_csv('/Users/ced/Documents/PolyHacks/Data/sector_Data/weekly_max_drawdown_by_sector.csv')
weekly_volatility_by_sector.to_csv('/Users/ced/Documents/PolyHacks/Data/sector_Data/weekly_volatility_by_sector.csv')
