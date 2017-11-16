import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import pickle
import pprint

def BollingerBand(ticker):
    #read the csv files
    df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), parse_dates=True, index_col=0)

    #get 20 days moving average
    df['20ma'] = df['Adj Close'].rolling(window=20, min_periods=0).mean()

    #get the standard deviation
    df['std'] = df['Adj Close'].rolling(window=20, min_periods=0).std()

    #get upperband
    df['upperband'] = df['20ma'].add((2 * df['std']))

    #get lowerband
    df['lowerband'] = df['20ma'].sub((2*df['std']))

    #get the calculation of h as bollinger band width
    df['h'] = ((df['upperband'].sub(df['lowerband'])).truediv(df['lowerband']))

    #set the BollingerBand.buy to empty list
    BollingerBand.buy=[]

    #set BollingerBand.t to false, so it's easier to print the qualified tickers
    BollingerBand.t= False


    print("---------------------------------------------------------------------")

    #if the h in yesterday data is more than 0.01 and less than 0.15. we buy the stock
    if (df['h'][-1] > 0.01 and df['h'][-1] < 0.15).any():
        print("{} is a buy according to bollinger bands".format(ticker))
        BollingerBand.buy.append(ticker)
        BollingerBand.t=True
        print(df.tail(3))

    #if it is more than 0.15 we don't want to buy
    elif(df['h'][-1] > 0.15).any():
        print("{} is a no buy according to bollinger bands".format(ticker))
    else:
        print("{} nothing goods here".format(ticker))

    print("---------------------------------------------------------------------")

# BollingerBand('ADRO')
def main():
    #load ticker data
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

        accuracies = []
        pp = pprint.PrettyPrinter(indent=4)
        for count,ticker in enumerate(tickers):

            tick = BollingerBand(ticker)
            # this is done in order to print BollingerBand.buy
            if (BollingerBand.t == True):
                accuracies.append(ticker)

        #use pprint to 'beautify' the output
        # f = open('workfile','w')
        # f.write(accuracies)
        pp.pprint(accuracies)
        # f.close()


main()
