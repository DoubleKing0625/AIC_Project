from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree


class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        # self.clf = RandomForestClassifier(n_estimators=2, max_leaf_nodes=2, random_state=61)
        # self.clf = XGBClassifier(learning_rate =0.1, n_estimators=26, max_depth=3, min_child_weight=5, gamma=0, subsample=0.5, colsample_bytree=0.9)
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
