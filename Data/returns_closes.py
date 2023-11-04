import pandas as pd

closes = pd.read_csv("Data/merged/merged_adjusted_close.csv", index_col=0, parse_dates=True)
returns = closes.pct_change().dropna(axis=0)

# Select columns that are not timestamps
non_timestamp_cols = [col for col in returns.columns if not isinstance(col, pd.Timestamp)]

# Calculate percentage change for non-timestamp columns
returns_non_timestamp = returns[non_timestamp_cols].pct_change().dropna(axis=0)

# Divide non-timestamp columns by 100
returns_non_timestamp[non_timestamp_cols] = returns_non_timestamp[non_timestamp_cols].div(100) + 1

# Export as CSV
returns_non_timestamp.to_csv("Data/merged/returns_non_timestamp.csv")
