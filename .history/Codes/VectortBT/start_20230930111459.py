import vectorbt 
import vectorbt as vbt
import numpy as np
import pandas
from datetime import datetime, timedelta    
import yfinance as yf

start = '2019-01-01 UTC'  # crypto is in UTC
end = '2021-01-01 UTC'
amzn_Price = vbt.YFData.download('TSLA', start=start, end=end).get('Close')
print(type(amzn_Price))
fast_ma = vbt.MA.run(amzn_Price,10)
slow_ma = vbt.MA.run(amzn_Price, 50)

rsi    = vbt .RSI.run(amzn_Price,10, short_name='rsi')

entries = fast_ma.ma_crossed_above(slow_ma) & (rsi.rsi_above(50))
exits = slow_ma.ma_crossed_above(fast_ma) & (rsi.rsi_below(50))

pf=vbt.Portfolio.from_signals(amzn_Price, entries, exits, fees=0.001, freq='d')
print(pf.total_profit())
print(pf.stats())