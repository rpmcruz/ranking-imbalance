#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from utils import load_csv
import numpy as np
import os
import shutil

try:
    shutil.rmtree('../out')
except:
    pass
try:
    os.mkdir('../out')
    os.mkdir('../out/train')
    os.mkdir('../out/test')
    os.mkdir('../out/predictions')
    os.mkdir('../out/scores')
except:
    pass

for f in os.listdir('../data'):
    print f
    X, y = load_csv(os.path.join('../data', f))
    #if len(y) > 1000:  # TEMP: faster train debug
    #    continue

    #for fold, (tr_index, ts_index) in enumerate(StratifiedKFold(y, 5, True)):
    for fold, (tr_index, ts_index) in enumerate(
            StratifiedShuffleSplit(y, 40, 0.2)):
        out = '../out/train/%s-fold%02d.csv' % (f[:-4], fold+1)
        data = np.c_[X[tr_index], y[tr_index]]
        np.savetxt(out, data, delimiter=',')

        out = '../out/test/%s-fold%02d.csv' % (f[:-4], fold+1)
        data = np.c_[X[ts_index], y[ts_index]]
        np.savetxt(out, data, delimiter=',')
