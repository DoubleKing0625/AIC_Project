from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pandas import Series, DataFrame
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def compute(Y, Y_p):
    tmp = Y_p[Y == 1]
    tp = tmp[tmp == 1].size
    fn = tmp[tmp == 0].size

    tmp = Y_p[Y == 0]
    fp = tmp[tmp == 1].size
    tn = tmp[tmp == 0].size

    score = (tp + tn) * 1.0 / (tp + fn + fp + tn)
    rappel = tp * 1.0 / (tp + fn)
    precision = tp * 1.0 / (tp + fp)

    return score, rappel, precision

def compute_model_logistic(X_train, Y_train, X_test, Y_test):
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_p = Series(lr.predict(X_test))
    score, rappel, precision = compute(Y_test, Y_p)

    learn_task_myself = []

    learn_task_myself.append(score)
    learn_task_myself.append(rappel)
    learn_task_myself.append(precision)

    learn_task_sklearn = []

    learn_task_sklearn.append(accuracy_score(Y_test, Y_p))
    learn_task_sklearn.append(recall_score(Y_test, Y_p))
    learn_task_sklearn.append(precision_score(Y_test, Y_p))

    return learn_task_myself, learn_task_sklearn

def compute_model_randomforest(X_train, Y_train, X_test, Y_test, n_estimators, max_depth):
    rf = RandomForestClassifier(n_estimators= n_estimators, max_depth=max_depth)
    rf.fit(X_train, Y_train)
    Y_p = Series(rf.predict(X_test))
    score, rappel, precision = compute(Y_test, Y_p)

    learn_task_myself = []

    learn_task_myself.append(score)
    learn_task_myself.append(rappel)
    learn_task_myself.append(precision)

    learn_task_sklearn = []

    learn_task_sklearn.append(accuracy_score(Y_test, Y_p))
    learn_task_sklearn.append(recall_score(Y_test, Y_p))
    learn_task_sklearn.append(precision_score(Y_test, Y_p))

    return learn_task_myself, learn_task_sklearn

def fine_tune(X, Y):
    param_test = {
        'n_estimators': range(10, 20, 2),
        'max_depth': range(2, 10, 2)
    }
    gsearch = GridSearchCV(estimator= RandomForestClassifier() ,param_grid = param_test, scoring = 'accuracy')
    gsearch.fit(X, Y)

    return gsearch.best_params_['n_estimators'], gsearch.best_params_['max_depth']