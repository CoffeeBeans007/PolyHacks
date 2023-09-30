import vectorbt 
import vectorbt as vbt
import numpy as np
import pandas
from datetime import datetime, timedelta    
import yfinance as yf

start='2019-01-01 UTC'
end=' 2020-01-01 UTC'

amzn_Price = vectorbt.YFData.download('AMZN', start=start, end=end)

fast_ma=vectorbt.MA.run(amzn_Price,10, short_name='fast')
slow_ma=vectorbt.MA.run(amzn_Price,50, short_name='slow')

rsi    = vbt .RSI.run(amzn_Price,10, short_name='rsi')

entries = fast_ma.crossed_above(slow_ma) & (rsi < 50)
exits = slow_ma.crossed_above(fast_ma) & (rsi > 50)