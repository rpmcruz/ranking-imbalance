# -*- coding: utf-8 -*-

# Extra models.


# One class SVM; considering only y==1

from sklearn.svm import OneClassSVM


class MyOneClassSVM(OneClassSVM):
    def fit(self, X, y):
        OneClassSVM.fit(self, X[y == 1])
        return self

    def predict_proba(self, X):
        return self.decision_function(X)

    def predict(self, X):
        return (OneClassSVM.predict(self, X)+1)/2


# LinearSVC does not have a predict_proba(): use decision_function()!

from sklearn.svm import SVC, LinearSVC


class MySVC(SVC):
    def predict_proba(self, X):
        return self.decision_function(X)

class MyLinearSVC(LinearSVC):
    def __init__(self, C=1., fit_intercept=True, class_weight=None):
        LinearSVC.__init__(self, C=C, fit_intercept=fit_intercept,
            class_weight=class_weight,
            penalty='l1', tol=1e-3, dual=False)  # like SVC

    def predict_proba(self, X):
        return self.decision_function(X)


# AdaBoost trained with a starting balanced distribution (Vázequez, 2004)
from utils import balanced_weights
from sklearn.ensemble import AdaBoostClassifier


class AdaBoostClassifierB(AdaBoostClassifier):
    def __init__(self):
        AdaBoostClassifier.__init__(self)

    def fit(self, X, y):
        w = balanced_weights(y)
        return AdaBoostClassifier.fit(self, X, y, w)
