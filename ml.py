from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE

from data import DataManager
from sklearn.metrics import roc_auc_score

import numpy as np

def mean_result(cnt=10):
    def decorator(function):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(cnt):
                results.append(function(*args, **kwargs))
            return np.mean(results)
        return wrapper
    return decorator

def rfe_feature_selection(data_manager: DataManager, classifier, step=1, fts=10):
    train_features, _, train_labels, _ = data_manager.train_split()
    rfe_method = RFE(
        classifier,
        step=1,
        n_features_to_select=fts
    )
    rfe_method.fit(train_features, train_labels)
    s = rfe_method.get_support()
    return list([data_manager.get_tested_features()[i] for i,v in enumerate(s) if v])

@mean_result()
def logistic_regression(data_manager: DataManager):
    train_features, test_features, train_labels, test_labels = data_manager.train_split()
    lr = LogisticRegression(solver='liblinear')
    lr.fit(train_features, train_labels)

    y_pred_prob = lr.predict_proba(test_features)[:, 1]
    return roc_auc_score(test_labels, y_pred_prob)

@mean_result()
def knn(data_manager: DataManager, neighbors=300):
    train_features, test_features, train_labels, test_labels = data_manager.train_split()
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(train_features, train_labels)

    y_pred_prob = knn.predict_proba(test_features)[:, 1]
    return roc_auc_score(test_labels, y_pred_prob)

@mean_result()
def gnb(data_manager: DataManager):
    train_features, test_features, train_labels, test_labels = data_manager.train_split()
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)

    y_pred_prob = gnb.predict_proba(test_features)[:, 1]
    return roc_auc_score(test_labels, y_pred_prob)

@mean_result()
def random_forest(data_manager: DataManager, estimators=100):
    train_features, test_features, train_labels, test_labels = data_manager.train_split()
    rf = RandomForestClassifier(n_estimators=estimators)
    rf.fit(train_features, train_labels)

    y_pred_prob = rf.predict_proba(test_features)[:, 1] 
    return roc_auc_score(test_labels, y_pred_prob)

@mean_result(5)
def mlp_classifier(data_manager: DataManager, hidden_layers=(10, 20, 10)):
    train_features, test_features, train_labels, test_labels = data_manager.train_split()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers,
                        activation='logistic',
                        solver='adam',
                        max_iter=1000, random_state=None)
    mlp.fit(X_train, train_labels)

    y_pred_prob = mlp.predict_proba(X_test)[:, 1]
    return roc_auc_score(test_labels, y_pred_prob)