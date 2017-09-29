# -*- coding: utf-8 -*-

import sklearn.cross_validation
import sklearn.metrics
from sklearn.base import clone
import numpy as np
import itertools
import time
import os


def test(names, models, k=4):
    for filename in sorted(os.listdir('../../data')):
        file = os.path.join('../../data', filename)
        with open(file, 'r') as f:
            counts = [f.readline().count(x) for x in (' ', ',')]
            delimiter = max(zip((' ', ','), counts), key=lambda x: x[1])[0]

        X = np.loadtxt(file, delimiter=delimiter)
        y = X[:, -1]
        X = X[:, 0:-1]
        filename = filename[:-4]  # truncate name

        # undersample
        idx = np.random.choice(np.arange(len(X)), min(len(X), 500), False)
        X = X[idx]
        y = y[idx]

        for i, (n, m) in enumerate(itertools.izip(names, models)):
            s = [0]*3
            tic = time.time()
            if k == 1:
                folds = sklearn.cross_validation.StratifiedShuffleSplit(y, k, 0.2)
            else:
                folds = sklearn.cross_validation.StratifiedKFold(y, k)
            for tr, ts in folds:
                m2 = clone(m).fit(X[tr], y[tr])
                yp = m2.predict(X[ts])
                s[0] += sklearn.metrics.f1_score(y[ts], yp)/k
                s[1] += (np.sum(np.logical_and(y[ts] == 0, yp == 0))/float(np.sum(y == 1)))/k
                s[2] += (np.sum(np.logical_and(y[ts] == 1, yp == 1))/float(np.sum(y == 0)))/k
            toc = time.time()
            print '%25s %-15s %.4f %.4f %.4f (%5.2fs)' % (filename, n, s[0], s[1], s[2], (toc-tic)/k)
        print


def twotest(m1, m2):
    test(('m1', 'm2'), (m1, m2))
