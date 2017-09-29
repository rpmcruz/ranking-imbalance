#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.base import clone
from extras import MyLinearSVC
from ranksvm import RankSVM
from smote import SMOTE
from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import os

try:
    os.makedirs('../../out/imbalances')
except:
    pass

# We are not doing CV. CV is changing scores by little, so it
# is not visible in the graphic.

models = (
    ('RankSVM', RankSVM()),
    ('SVM b', MyLinearSVC(class_weight='balanced')),
    ('SVM Linear SMOTE', SMOTE(MyLinearSVC(), False)),
)
NORMALIZE = True


filename = 'spambase'
data = np.loadtxt('../../data/%s.csv' % filename, delimiter=',')
X = data[:, 0:-1]
y = data[:, -1]
N = len(X)
N1 = np.sum(y)
print 'N1 = %d' % N1

if NORMALIZE:
    mu = np.mean(X, 0)
    sd = np.std(X, 0)
    X = (X - mu) / sd


def run_imbalance(IR):
    exists = False
    for name, _ in models:  # see if any model is missing
        outname = '../../out/imbalances/%s-%.05f.txt' % (name, IR)
        exists = exists or os.path.exists(outname)
    if exists:
        return

    undersample = int(N1 - IR*N)
    n1 = N1 - undersample
    print 'IR: %.5f - n1 = %d' % (IR, n1)

    if n1 <= 0:
        return  # ignore cases when no classes exist

    # force unbalance !
    # we reduce the first n1 observations -- the data folds have already
    # been shuffled, so order is unimportant
    y_ = np.array(y, int)
    #y_[ytr == 1][0:(N1-n1)] = 0  # does not work
    y_[np.where(y)[0][0:(N1-n1)]] = 0

    k = use_folds
    s_avg = [0]*len(models)
    for tr, ts in StratifiedKFold(y_, k, True):
        for i, (name, m) in enumerate(models):
            m = clone(m).fit(X[tr], y_[tr])
            yp = m.predict(X[ts])
            s_avg[i] += f1_score(y_[ts], yp) / float(use_folds)
    for i, (name, _) in enumerate(models):
        outname = '../../out/imbalances/%s-%.05f.txt' % (name, IR)
        with open(outname, 'w') as f:
            # repr does not lose precision:
            # http://stackoverflow.com/questions/3481289/converting-a-
            # python-float-to-a-string-without-losing-precision
            f.write(repr(s_avg[i]))


use_folds = 40  # maximum=40 (may be less)
iterations = 10
min_IR = N1/float(N)
max_IR = 0.01
IRs = np.linspace(min_IR, max_IR, iterations)
use_processes = 3  # avoid many cpus or out of memory

import multiprocessing
print 'Using %d processes, %d iterations, %d folds' % (
    use_processes, iterations, use_folds)
p = multiprocessing.Pool(use_processes)
p.map(run_imbalance, IRs)
