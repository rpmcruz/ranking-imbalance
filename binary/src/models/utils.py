# -*- coding: utf-8 -*-

import numpy as np

# Utilities for our models.


def balanced_class_weights(y):
    bincount = np.asarray((np.sum(y == 0), np.sum(y == 1)))
    return len(y) / (2. * bincount)


def balanced_weights(y):
    wclass = balanced_class_weights(y)
    return np.asarray([wclass[i] for i in y])


def choose_threshold(s, y):
    #return np.amin(s[y == 1])
    #return np.median(s)
    si = np.argsort(s)
    s = s[si]
    y = y[si]
    #sy = sorted(zip(s, y))
    #s = [x for x, _ in sy]
    #y = [x for _, x in sy]

    maxF1 = -np.inf
    bestTh = 0

    for i in xrange(1, len(y)):
        if y[i] != y[i-1]:
            TP = np.sum(y[i:] == 1)
            FP = np.sum(y[i:] == 0)
            FN = np.sum(y[:i] == 1)
            F1 = (2.*TP)/(2.*TP+FN+FP+1e-10)
            if F1 > maxF1:
                maxF1 = F1
                bestTh = (s[i]+s[i-1])/2.

    return bestTh
