from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


class Modeling:
    """

    """

    def __init__(self, dataset: pd.DataFrame, target_col: str, *kwargs):
        self.dataset = dataset
        self.target_col = target_col
        self.params = kwargs

    def split_train_test(self):
        X = self.dataset.drop(columns=[self.target_col])
        y = self.dataset[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, self.params)
        return X_train, X_test, y_train, y_test

    def xgboost(self):
        model = xgb.XGBClassifier(self.params)
        X_train, X_test, y_train, y_test = self.split_train_test()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_test, y_pred

    def random_forest(self):
        model = RandomForestClassifier(self.params)
        X_train, X_test, y_train, y_test = self.split_train_test()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_test, y_pred

    def logistic_regression(self):
        model = LogisticRegression(self.params)
        X_train, X_test, y_train, y_test = self.split_train_test()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_test, y_pred


def roc_curve(y_test: pd.Series, y_pred: pd.Series):
    """

    """

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    AUC = metrics.auc(fpr, tpr)
    print('AUC: ', AUC)


def showing(y_test: pd.Series, y_pred: pd.Series) -> None:
    """

    """

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')
    cm = metrics.classification_report(y_test, y_pred)
    print('Accuracy: {:.2f}\n'
          'Precision: {:.2f}\n'
          'Recall: {:.2f}\n'
          'f1_score: {:.2f}\n'
          'cm:\n{}'.format(accuracy, precision, recall, f1_score, cm))

    if isinstance(y_test.iloc[0], int) | isinstance(y_test.iloc[0], float):
        roc_curve(y_test, y_pred)

