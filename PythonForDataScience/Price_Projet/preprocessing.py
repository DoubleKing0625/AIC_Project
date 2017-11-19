'''
def popularized()
def getdumpling()
def pre_processing()
'''

import pandas as pd
from sklearn.model_selection import ShuffleSplit
import numpy as np


def add_prob(df, newcol, col, n):
    df[newcol] = df[col].count() * 1.0 / n
    return df

def popularized(df, col_to_popularize):
    n = len(df)
    for feature in col_to_popularize:
        pop_feature = feature + "_prop"
        df = df.groupby(feature).apply(add_prob, pop_feature, feature,n)
        del df[feature]
    return df

def get_dummy(df, col_to_dummy):
    for dum in col_to_dummy:
        split_temp = pd.get_dummies(df[dum], prefix=dum)
        for col in split_temp:
            df[col] = split_temp[col]
        del df[dum]
    return df

def pre_processing(df, col_to_popularize = [], col_to_dummy = []):
    df = popularized(df, col_to_popularize)
    df = get_dummy(df, col_to_dummy)
    return df

def pre_processing_target(df):
    return df['MT_VNT_EUR_VOY']

def pre_processing_train(df):
    tmp = df.copy()
    del tmp['MT_VNT_EUR_VOY']
    return tmp

def extract_data(x, target):
    sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, test_index in sss.split(x, target):
        X = x[test_index]
        y = target[test_index]

    return X, y

def split_data(x, target):
    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(x, target):
        X_train = x[train_index]
        X_test = x[test_index]
        y_train = target[train_index]
        y_test = target[test_index]

    return X_train, y_train, X_test, y_test