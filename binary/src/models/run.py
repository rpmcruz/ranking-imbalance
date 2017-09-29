#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Run python models.

from sklearn.base import clone
import os
import sys
import numpy as np
from smote import SMOTE, MSMOTE
from metacost import MetaCost

#models = ( (name, function to create object, create estimator) )

from extras import MyOneClassSVM
from extras import MyLinearSVC
from ranksvm import RankSVM
from jaimesvm import JaimeSVM

base = None

# NOTE: LinearSVC does not have predict_proba(), what we do is to overload it
# in our validation class, CCV -- that's why we use MyLinearSVC.

# It is important to overload and return best_estimator_, otherwise MetaCost
# will try to run cross validation again!

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.grid_search import GridSearchCV

class C_GridSearch(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator
        self.classes_ = (0, 1)

    def fit(self, X, y):
        m = self.estimator
        return GridSearchCV(m, {'C': np.logspace(-2, 2, 5)}, 'f1', cv=5). \
            fit(X, y).best_estimator_

models1 = (
    ('RankSVM', lambda: C_GridSearch(RankSVM()), None),
    ('SVM Linear', lambda: C_GridSearch(MyLinearSVC()), None),
    ('SVM Linear b', lambda: C_GridSearch(MyLinearSVC(class_weight='balanced')), None),
    ('SVM Linear SMOTE', lambda: SMOTE(base), 1),
    ('SVM Linear MSMOTE', lambda: MSMOTE(base), 1),
    ('SVM Linear MetaCost', lambda: MetaCost(base, 'balanced', False), 1),
    #('JaimeSVM', lambda: C_GridSearch(JaimeSVM()), None),
)

from neuralnet import NeuralNet, RankNet

class Hidden_GridSearch(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator
        self.classes_ = (0, 1)

    def fit(self, X, y):
        hmin = X.shape[1]
        hmax = X.shape[1]*2
        nbr = min(5, hmax-hmin+1)

        m = clone(self.estimator)
        m.maxit = 100
        cv = GridSearchCV(m,
            {'hidden_nodes': np.linspace(hmin, hmax, nbr).astype(int)}, 'f1',
            cv=5, refit=False).fit(X, y)
        print 'best params:', cv.best_params_

        self.estimator.hidden_nodes = cv.best_params_['hidden_nodes']
        self.estimator.maxit = 1000
        return self.estimator.fit(X, y)

models2 = (
    ('RankNet', lambda: Hidden_GridSearch(RankNet(0)), None),
    ('NeuralNet', lambda: Hidden_GridSearch(NeuralNet(0, False)), None),
    ('NeuralNet b', lambda: Hidden_GridSearch(NeuralNet(0, True)), None),
    ('NeuralNet SMOTE', lambda: SMOTE(base), 0),
    ('NeuralNet MSMOTE', lambda: MSMOTE(base), 0),
    ('NeuralNet MetaCost', lambda: MetaCost(base, 'balanced', False), 0),
)


from adaboost import AdaBoost, RankBoost

models3 = (
    ('RankBoost', lambda: RankBoost(50), None),
    ('AdaBoost', lambda: AdaBoost(50), None),
    ('AdaBoost b', lambda: AdaBoost(50, balanced=True), None),
    ('AdaBoost SMOTE', lambda: SMOTE(AdaBoost(50)), None),
    ('AdaBoost MSMOTE', lambda: MSMOTE(AdaBoost(50)), None),
    ('AdaBoost MetaCost', lambda: MetaCost(AdaBoost(50), 'balanced', False), None),
)

from gbrank import GBRank
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

"""
estimators_used = 0

class GBRank2(GBRank):
    def fit(self, X, y):
        GBRank.fit(self, X, y)
        global estimators_used
        estimators_used = len(self.g)
        return self

class RandomForest2(RandomForestClassifier):
    def fit(self, X, y):
        self.set_params(n_estimators=estimators_used)
        RandomForestClassifier.fit(self, X, y)
        return self
"""

n_estimators = 10
max_depth = 2
models4 = (
    ('GBRank', lambda: GBRank(DecisionTreeRegressor(max_depth=max_depth), n_estimators), None),
    ('Random Forest', lambda: RandomForestClassifier(n_estimators, max_depth=max_depth), None),
    ('Random Forest b', lambda: RandomForestClassifier(n_estimators, max_depth=max_depth, class_weight='balanced'), None),
    ('Random Forest SMOTE', lambda: SMOTE(RandomForestClassifier(n_estimators, max_depth=max_depth)), None),
    ('Random Forest MSMOTE', lambda: MSMOTE(RandomForestClassifier(n_estimators, max_depth=max_depth)), None),
    ('Random Forest MetaCost', lambda: MetaCost(RandomForestClassifier(n_estimators, max_depth=max_depth), 'balanced', False), None),
)

SMALL_FILES = False
models = models1 + models2 + models3 + models4

# make sure the directories exist
for name, _, _ in models:
    try:
        os.makedirs(os.path.join('../../out/predictions', name))
        os.makedirs(os.path.join('../../out/scores', name))
    except:
        pass


import warnings
warnings.filterwarnings('ignore')
# We will ignore sklearn warnings because of this sklearn bug:
# https://github.com/scikit-learn/scikit-learn/issues/2586
# It issues false warnings for F1-score when recall or prec=0.
# NOTE: we still show important errors and exceptions.

import time
files = os.listdir('../../out/train')
files.sort()

# run models for each train & test
def run_file(file):
    print file
    data = np.loadtxt(os.path.join('../../out/train', file), delimiter=',')
    _X = data[:, 0:-1]
    y = data[:, -1]

    #if SMALL_FILES and len(X) > 2000:
    #    return

    data = np.loadtxt(os.path.join('../../out/test', file), delimiter=',')
    _Xp = data[:, 0:-1]

    base_models = [None] * len(models)
    times = [0] * len(models)
    for modeli, (name, model, base_model) in enumerate(models):
        if (name, model, base_model) in models1:  # normalize
            mu = np.mean(_X, 0)
            sd = np.std(_X, 0) + 1e-8
            X = (_X - mu) / sd
            Xp = (_Xp - mu) / sd
        else:
            X = _X
            Xp = _Xp

        print name
        predsname = os.path.join('../../out/predictions', name, file)
        scoresname = os.path.join('../../out/scores', name, file)
        create = not os.path.exists(predsname) or not \
            os.path.exists(scoresname)
        if create:
            tic = time.time()
            try:
                global base
                if base_model is not None:
                    if base_models[base_model] is None:
                        # this can happen when run is aborted
                        base_models[base_model] = models[base_model][1](). \
                            fit(X, y)
                    base = clone(base_models[base_model])
                m = model().fit(X, y)
                base_models[modeli] = m
            except ValueError as err:
                print '%s (%s %s)' % (err.args[0], file, name)
            else:
                yp = m.predict(Xp)
                ys = m.predict_proba(Xp)
                np.savetxt(predsname, yp, '%d')
                np.savetxt(scoresname, ys)
            times[modeli] = time.time() - tic
            #print times[modeli]

    # print the times at the end to avoid processes talking over each other
    progress = len(os.listdir(os.path.join(
        '../../out/predictions', models[0][0]))) / float(len(files))
    print '%-25s times (%3d%%)' % (file, progress*100)
    for modeli, (name, model, base_model) in enumerate(models):
        print '\t%19s: %5.2fm' % (name, times[modeli] / 60.)

import multiprocessing
use_processors = multiprocessing.cpu_count()
print 'Using %d processes' % use_processors
p = multiprocessing.Pool(use_processors)
p.map(run_file, files)
