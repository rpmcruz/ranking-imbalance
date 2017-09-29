# -*- coding: utf-8 -*-

# Ranking models.

from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.svm import SVC
from extras import MyLinearSVC
from utils import choose_threshold
from ranksvm import preprocess
import numpy as np
import itertools


class JaimeSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1):
        super(JaimeSVM, self).__init__()
        self.C = C
        self.classes_ = (0, 1)

    def fit(self, X, y):
        # we only instantiate it here, so that sklearn can do its
        # GridSearch validation magic with get_ and set_params()
        self.estimator = MyLinearSVC(C=self.C)
        #self.estimator = SVC(C=self.C, kernel='linear')

        dX, dy = preprocess(X, y)
        self.estimator.fit(dX, dy)
        return self

    def predict_proba(self, X):
        return self.estimator.decision_function(X)

    def predict(self, X):
        return self.estimator.predict(X)
