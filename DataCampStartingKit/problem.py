import os
import sys
import numpy as np
import pandas as pd
import rampwf as rw
import pickle
import gzip
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'test'
# _target_column_name = 'target'
_prediction_label_names = [0,1,2,3,4,5,6,7,8,9]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
    #rw.score_types.NormalizedGini(name='ngini', precision=3),
    #rw.score_types.ROCAUC(name='auc', precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
    #rw.score_types.NegativeLogLikelihood(name='nll', precision=3),
]


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin-1')
    fo.close()
    return dict

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.5, random_state=57)
    return cv.split(X, y)

# --------------- X is DataFrame, y is numpy ---------------- #
def get_train_data(path='.'):
    f = gzip.open('./data/mnist.pkl.gz', 'rb')

    if (sys.version_info.major == 2):
        train_set, valid_set, test_set = pickle.load(f)  # compatibility issue between python 2.7 and 3.4
    else:
        train_set, valid_set, test_set = pickle.load(f,
                                                     encoding='latin-1')  # compatibility issue between python 2.7 and 3.4
    f.close()

    # Shuffle
    train_X = pd.DataFrame(train_set[0])
    train_y = pd.DataFrame(train_set[1]).values

    return train_X, train_y



def get_test_data(path='.'):
    f = gzip.open('./data/mnist.pkl.gz', 'rb')

    if (sys.version_info.major == 2):
        train_set, valid_set, test_set = pickle.load(f)  # compatibility issue between python 2.7 and 3.4
    else:
        train_set, valid_set, test_set = pickle.load(f,
                                                     encoding='latin-1')  # compatibility issue between python 2.7 and 3.4
    f.close()

    # Shuffle
    test_X = pd.DataFrame(valid_set[0])
    test_y = pd.DataFrame(valid_set[1]).values

    return test_X, test_y

'''
def save_submission(y_pred, data_path='.', output_path='.', suffix='test'):
    if 'test' not in suffix:
        return  # we don't care about saving the training predictions
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        sample_df = pd.read_csv(os.path.join(
            data_path, 'data', 'sample_submission.csv'))
    else:
        sample_df = pd.read_csv(os.path.join(
            data_path, 'kaggle_data', 'sample_submission.csv'))
    df = pd.DataFrame()
    df['id'] = sample_df['id']
    df['target'] = y_pred[:, 1]
    output_f_name = os.path.join(
        output_path, 'submission_{}.csv'.format(suffix))
    df.to_csv(output_f_name, index=False)
'''