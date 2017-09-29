# -*- coding: utf-8 -*-

# Ranking models: this is for the binary care.

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


def preprocess(X, y):
    klasses = np.unique(y)
    count = np.bincount(y)[klasses]

    # N: number of obs from all combinations
    # K: number of combinations
    N = 0
    for ki in range(len(klasses)):
        for kj in range(ki+1, len(klasses)):
            N += 2*count[ki]*count[kj]
    K = len(klasses)*(len(klasses)-1)/2

    dX = np.zeros((N, X.shape[1]), X.dtype)
    dy = np.zeros(N)
    w = np.zeros(N)
    i = 0
    for ki in range(len(klasses)):
        for kj in range(ki+1, len(klasses)):
            k1 = klasses[ki]
            k2 = klasses[kj]

            n = 2*count[ki]*count[kj]
            _w = N/(K*n)

            for X1 in X[[y == k1]]:
                for X2 in X[[y == k2]]:
                    dX[i] = X1 - X2
                    dy[i] = +1
                    w[i] = _w
                    i += 1
                    dX[i] = X2 - X1
                    dy[i] = -1
                    w[i] = _w
                    i += 1
    assert i == N
    return dX, dy, w


class RankSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        super(RankSVM, self).__init__()
        self.model = model
        if model is None:
            from sklearn.svm import LinearSVC
            self.model = LinearSVC(fit_intercept=False, penalty='l1', tol=1e-3,
                                   dual=False)

    def fit(self, X, y, sample_weight=None):
        dX, dy, w = preprocess(X, y)
        self.model.fit(dX, dy, w)
        if isinstance(self.model, GridSearchCV):
            m = self.model.best_estimator_
        else:
            m = self.model
        self.coefs = m.coef_[0]
        return self

    def predict_proba(self, X):
        # these are not probabilities, but I overloaded this function because
        # it makes it nicer to have a common interface for the ROC curves
        return -np.sum(self.coefs*X, 1)

    def decision_function(self, X):
        return self.predict_proba(X)
