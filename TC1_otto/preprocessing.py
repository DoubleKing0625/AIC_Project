import pandas as pd

def loadTrainSet():
    train = pd.read_csv('./data/train.csv')
    train_y = train['target'].apply(lambda s: int(s[-1:])).values
    train_X = train.drop('id', axis=1)
    train_X = train_X.drop('target', axis=1)
    
    train_set = []
    train_set.append(train_X[:50000])
    train_set.append(train_y[:50000])

    valid_set = []
    valid_set.append(train_X[50000:])
    valid_set.append(train_y[50000:])
    return train_set, valid_set

def loadTest():
    test = pd.read_csv('./data/test.csv')
    test_X = test.drop('id', axis=1)
    return test_X