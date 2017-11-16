import pandas as pd
import datetime as dt
import talib
from pandas.stats.moments import ewma

def compute_macd(ticker):
    df = pd.read_csvdf = pd.read_csv('stock_dfs/{}.csv'.format(ticker), parse_dates=True, index_col=0)

    df['ewma12'] = ewma(df['Adj Close'], span=12)
    df['ewma26'] = ewma(df['Adj Close'], span=26)
    df['MACD'] = df['ewma12'].sub(df['ewma26'])
    df['macdsignal'] = ewma(df['MACD'], span=9)
    df['histogram'] = df['MACD'].sub(df['macdsignal'])

    if(df['histogram'][-1] > 0 and df['MACD'][-1] > 0):
        print('{} MACD is bullish'.format(ticker))
    elif(df['histogram'][-1] < -0.15):
        print('{} MACD is bearish'.format(ticker))
    else:
        print("macd is aight")
    print('ok')
    print(df.tail(1))
compute_macd('')
