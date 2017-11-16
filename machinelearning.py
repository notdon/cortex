from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from statistics import mean


def process_data_for_labels(ticker):
    hm_days = 5
    df = pd.read_csv('sp500joined.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > 0.027:
            return 1
        if col < -0.025:
            return -1

    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                    df['{}_1d'.format(ticker)],
                                    df['{}_2d'.format(ticker)],
                                    df['{}_3d'.format(ticker)],
                                    df['{}_4d'.format(ticker)],
                                    df['{}_5d'.format(ticker)],
                                    # df['{}_6d'.format(ticker)],
                                    # df['{}_7d'.format(ticker)],
                                    # df['{}_8d'.format(ticker)],
                                    # df['{}_9d'.format(ticker)],
                                    # df['{}_10d'.format(ticker)],
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
#extract_featuresets('SRIL')
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

# df = pd.read_csv('sp500joined.csv', index_col=0)
# tickers = df.columns.values.tolist()
# for ticker in tickers:
#     print("\n"+ticker + " \n")
#     do_ml(ticker)
with open("sp500tickers.pickle","rb") as f:
    tickers = pickle.load(f)

accuracies = []
for count,ticker in enumerate(tickers):

    # if count%10==0:
    #     print(count)

    accuracy = do_ml(ticker)
    accuracies.append(accuracy)
    print("\n{} accuracy: {}. Average accuracy:{}\n".format(ticker,accuracy,mean(accuracies)))
    print("------------------------------------------------------------")
