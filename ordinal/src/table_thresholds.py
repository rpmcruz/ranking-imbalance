from scipy.stats import kendalltau
from scores import maximum_mean_absolute_error
from sklearn.metrics import mean_absolute_error
from threshold_kelwin import decide_thresholds
from threshold import threshold_to_class
import numpy as np
import time
import os
import sys

kernel = 'linear'
dirname = '../data/30HoldoutOrdinalImbalancedDatasets'
datasets = os.listdir(dirname)

model = 'ranksvm-abs'


for dataset in datasets:
    print('** %s' % dataset)
    print()
    scores = np.zeros((3, 2))
    times = np.zeros(2)
    for fold in range(30):
        sys.stdout.write('\rfold %02d' % fold)
        sys.stdout.flush()

        filename = '../out/linear-%s--%s--%d--pp.csv' % (model, dataset, fold)
        pp = np.loadtxt(filename)
        filename = '%s/%s/matlab/test_%s.%d' % (
            dirname, dataset, dataset, fold)
        y = np.loadtxt(filename)[:, -1].astype(int)-1
        k = np.amax(y)+1
        for i, full in enumerate((False, True)):
            tic = time.time()
            ths = decide_thresholds(pp, y, k, 'absolute', full)
            yp = threshold_to_class(pp, ths)
            toc = time.time()
            times[i] += (toc-tic)/30
            scores[0, i] += mean_absolute_error(y, yp)/30
            scores[1, i] += maximum_mean_absolute_error(y, yp)/30
            scores[2, i] += kendalltau(y, yp)[0]/30
    sys.stdout.write('\r            \r')
    for i, full in enumerate((False, True)):
        print('%s took %f secs' % ('full' if full else 'kelwin', times[i]))
        print('\t mae: %f' % scores[0, i])
        print('\tmmae: %f' % scores[1, i])
        print('\t tau: %f' % scores[2, i])
        print()
