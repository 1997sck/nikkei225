import bs4 as bs
from collections import Counter
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import statistics
def save_nikkei225_tickers():
    payload = pd.read_html('https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/')
    table_0 = payload[0]
    df = table_0
    tickers_tmp = df['Code'].values.tolist()
    tickers = []
    for t in tickers_tmp:
        t1 = str(t) + '.T'
        tickers.append(t1)

    with open("nikkei225tickers.pickle","wb") as f:
        pickle.dump(tickers,f)


    return tickers

save_nikkei225_tickers()

def get_data_from_yahoo(reload_nikkei225=False):

    if reload_nikkei225:
        tickers = save_nikkei225_tickers()
    else:
        with open("nikkei225tickers.pickle","rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2015, 1, 1)
    end = dt.datetime(2016, 1, 31)

    for ticker in tickers[:10]:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, "yahoo", start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

##get_data_from_yahoo()

def compile_data_nikkei225():

    with open("nikkei225tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
            print(main_df.head())
        main_df.to_csv('nikkei225_joined_closes.csv')


compile_data_nikkei225()
def process_data_for_labels(ticker):
    hm_days = 7
    #df = pd.read_csv('nikkei225_joined_closes.csv', index_col=0)
    df = pd.read_csv('nikkei225_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))


    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker_name for ticker_name in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X,y,df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25)

    #clf = neighbors.KNeighborsClassifier()

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])


    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))
    print()
    print()
    return confidence

# from statistics import mean
#
# with open("sp500tickers.pickle","rb") as f:
#     tickers = pickle.load(f)
#
# accuracies = []
# for count,ticker in enumerate(tickers):
#
#     if count%10==0:
#         print(count)
#
#     accuracy = do_ml(ticker)
#     accuracies.append(accuracy)
#     print("{} accuracy: {}. Average accuracy:{}".format(ticker,accuracy,mean(accuracies)))
do_ml('6857.T')

#save_nifty50_tickers()
#get_data_from_yahoo_nifty50()
