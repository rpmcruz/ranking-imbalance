from sklearn.base import BaseEstimator, ClassifierMixin
from threshold_kelwin import decide_thresholds
import numpy as np


def choose_threshold_bin(s, y):
    bestTh = 0
    maxF1 = -np.inf
    for i in range(1, len(y)):
        if y[i] != y[i-1]:
            TP = np.sum(y[i:] == 1)
            FP = np.sum(y[i:] == 0)
            FN = np.sum(y[:i] == 1)
            F1 = (2.*TP)/(2.*TP+FN+FP+1e-10)
            if F1 > maxF1:
                maxF1 = F1
                bestTh = (s[i]+s[i-1])/2.
    return bestTh, maxF1

# left to right threshold

def choose_threshold_ltr(s, y):
    klasses = np.unique(y)

    ths = np.zeros(len(klasses)-1)
    for ki, k in enumerate(klasses[:-1]):
        ths[ki], _ = choose_threshold_bin(s, y <= k)
    return ths


# try thresholds from both sides

def choose_threshold_two(s, y):
    klasses = np.unique(y)

    ths = np.ones(len(klasses)-1)
    for ki, k in enumerate(klasses[:-1]):
        left = y <= k
        right = y > k
        left_th, left_f1 = choose_threshold_bin(s, left)
        right_th, right_f1 = choose_threshold_bin(s, right)
        if left_f1 > right_f1:
            ths[ki] = left_th
        else:
            ths[ki] = right_th
    return ths


def choose_threshold(s, y):
    return choose_threshold_ltr(s, y)


def threshold_to_class(s, ths):
    return np.sum(s >= ths[:, np.newaxis], 0, int)


class Threshold(BaseEstimator, ClassifierMixin):
    def __init__(self, model, strategy):
        self.model = model
        self.strategy = strategy

    def fit(self, X, y):
        # self.classes_ = np.unique(y)  # required by sklearn
        # for our datasets, this makes more sense:
        self.classes_ = np.arange(np.amax(y)+1, dtype=int)

        self.model.fit(X, y)
        s = self.model.predict_proba(X)

        # this class ensure that scores are ordered
        i = np.argsort(s)
        s = s[i]
        y = y[i]

        if self.strategy == 'ltr':
            self.ths = choose_threshold_ltr(s, y)
        elif self.strategy == 'between':
            self.ths = choose_threshold_two(s, y)
        else:
            self.ths = decide_thresholds(
                s, y, len(self.classes_), self.strategy)
        return self

    def predict(self, X):
        s = self.model.predict_proba(X)
        yp = threshold_to_class(s, self.ths)
        return yp

    def predict_proba(self, X):
        return self.model.predict_proba(X)
