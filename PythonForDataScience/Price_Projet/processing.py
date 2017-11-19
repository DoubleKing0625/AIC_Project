from sklearn.ensemble import RandomForestRegressor
from pandas import Series, DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


def compute_model_randomforest(X_train, Y_train, X_test, Y_test, n_estimators, max_depth):
    rf = RandomForestRegressor(n_estimators= n_estimators, max_depth=max_depth)
    rf.fit(X_train, Y_train)
    Y_p = Series(rf.predict(X_test))

    mse = mean_squared_error(Y_test, Y_p)
    rmse = mse ** 0.5
    return rmse

def fine_tune(X, Y):
    param_test = {
        'n_estimators': range(100, 300, 50),
        'max_depth': range(2, 10, 2)
    }
    gsearch = GridSearchCV(estimator= RandomForestRegressor() ,param_grid = param_test, scoring = 'neg_mean_squared_error')
    gsearch.fit(X, Y)

    return gsearch.best_params_['n_estimators'], gsearch.best_params_['max_depth']