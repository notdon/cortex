import pandas as pd
import datetime as dt
from stockstats import StockDataFrame
def rsi(ticker):
    df = StockDataFrame.retype(pd.read_csv('stock_dfs/{}.csv'.format(ticker), parse_dates=True, index_col=0))
    df['rsi_14']
    print(df.tail())
rsi('ADRO')
