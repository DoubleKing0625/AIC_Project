import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def pre_processing(x):
    if 'Survived' not in x.columns:
        raise ValueError("target column survived should belong to df")
    target = x['Survived']
    # x['title'] = x['Name'].map(lambda x: x.split(',')[1].split('.')[0])
    # x['surname'] = x['Name'].map(lambda x: '(' in x)
    x['Cabin'] = x['Cabin'].map(lambda x: x[0] if not pd.isnull(x) else -1)
    to_dummy = ['Pclass', 'Sex', 'Embarked', 'Cabin']
    for dum in to_dummy:
        split_temp = pd.get_dummies(x[dum], prefix = dum)
        for col in split_temp:
            x[col] = split_temp[col]
        del x[dum]
    x['Age'] = x['Age'].fillna(x['Age'].median())
    to_del = ["PassengerId", "Name", "Survived", "Ticket"]
    for col in to_del:
        del x[col]
    return x, target

#  we split the data, 80% is train-set, 20% is test_set
def split_data(x, target):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(x, target):
        X_train = x[train_index]
        X_test = x[test_index]
        y_train = target[train_index]
        y_test = target[test_index]

    return X_train, y_train, X_test, y_test