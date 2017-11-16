import bs4 as bs
import datetime as dt
import os
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from statistics import mean

style.use('ggplot')
# This method is used to to pull all the tickers of LQ45 from wikipedia using beautiful soup
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/LQ45')
    soup = bs.BeautifulSoup(resp.text,"lxml")
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        #there will be error when pulling stocks without this line
        mapping = str.maketrans(".","-")
        ticker = ticker.translate(mapping)
        tickers.append(ticker)
    # Read extra tickers from txt file
    for x in open('extra.txt','r').readlines():
        tickers.append(x.strip())
    # write to pickle
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)

    print(tickers)

    return tickers

# This method is used to get data from yahoo using pandas DataReader
def get_data_from_yahoo(reload_sp500=False):
    #just in case if pickle not written
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        #read the pickle file
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
    #create a directory for all the stocks data frame
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2016,12,1)
    end = dt.datetime(2017,5,5)

    #get all the tickers with .JK since we want pull data for Indonesian company
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker+".JK", 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

# we want to create a joined csv file for all the tickers and data in stock_dfs
def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)
    main_df= pd.DataFrame()

    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker},inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)

        if main_df.empty:
                main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    # print(main_df.head())
    main_df.to_csv('sp500joined.csv')

# This methods is used to visualize the correlation between all tickers
def visualize_data():
        df = pd.read_csv('sp500joined.csv')
        # df['TLKM'].plot()
        # plt.show()
        df_corr = df.corr()

        print(df_corr.head())

        data = df_corr.values
        fig = plt.figure()

        ax= fig.add_subplot(1,1,1)
        heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
        fig.colorbar(heatmap)

        ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

        ax.invert_yaxis()
        ax.xaxis.tick_top()
        column_lables = df_corr.columns
        row_labels = df_corr.index

        ax.set_xticklabels(column_lables)
        ax.set_yticklabels(row_labels)

        plt.xticks(rotation=90)
        heatmap.set_clim(-1,1)
        plt.tight_layout()
        plt.show()

# This method is used to process the data for label in Machine Learning
def process_data_for_labels(ticker):
    hm_days = 10
    df = pd.read_csv('sp500joined.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

# This method is used to classified if a stock is buy or sell feel free to change the requirement
# the requirement means the percentage of our expected profit. for example in
# requirement is 0.02 is 2% expected return at least between hm_days
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > 0.022:
            return 1
        if col < -0.01999:
            return -1

    return 0

# Feature sets for machine learning. remember to change the list
# depends on how many days to extract
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                    df['{}_1d'.format(ticker)],
                                    df['{}_2d'.format(ticker)],
                                    df['{}_3d'.format(ticker)],
                                    df['{}_4d'.format(ticker)],
                                    df['{}_5d'.format(ticker)],
                                    df['{}_6d'.format(ticker)],
                                    df['{}_7d'.format(ticker)],
                                    df['{}_8d'.format(ticker)],
                                    df['{}_9d'.format(ticker)],
                                    df['{}_10d'.format(ticker)],
                                    # df['{}_11d'.format(ticker)],
                                    # df['{}_12d'.format(ticker)],
                                    # df['{}_13d'.format(ticker)],
                                    # df['{}_14d'.format(ticker)],
                                    # df['{}_15d'.format(ticker)],
                                    # df['{}_16d'.format(ticker)],
                                    # df['{}_17d'.format(ticker)],
                                    # df['{}_18d'.format(ticker)],
                                    # df['{}_19d'.format(ticker)],
                                    # df['{}_20d'.format(ticker)],
                                    ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data Spread:', Counter(str_vals))

    df.fillna(0, inplace=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0,inplace=True)

    X = df_vals.values
    y= df['{}_target'.format(ticker)].values

    return X, y, df

# Do machine learning stuff
def do_ml(ticker):
    X, y, df, = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.25)

    #clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),('knn',neighbors.KNeighborsClassifier()),
    ('rfor', RandomForestClassifier())])


    clf.fit(X_train, y_train)

    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)
    predictions = clf.predict(X_test)

    print('Predicted spread:', Counter(predictions))

    return confidence

# Main function
def main():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

        accuracies = []
        for count,ticker in enumerate(tickers):

            # if count%10==0:
            # print(count)

            accuracy = do_ml(ticker)
            accuracies.append(accuracy)
            print("\n{} accuracy: {}. Average accuracy:{}\n".format(ticker,accuracy,mean(accuracies)))
            print("------------------------------------------------------------")



#
save_sp500_tickers()
get_data_from_yahoo()
compile_data()
max_loop = 99
for x in range(0, max_loop):
    print("\n This is new line", x)
    print("------------------------------------------------------------------------------------")
    main()
    print("------------------------------------------------------------------------------------")

