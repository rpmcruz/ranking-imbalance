import os
import numpy as np
from sklearn.svm import SVC, LinearSVC
from scipy.stats import kendalltau
from scores import maximum_mean_absolute_error
from sklearn.metrics import make_scorer, mean_absolute_error
from ranksvm import RankSVM
from mysvmlight import SVMLight
from mysvor import SVORIM, SVOREX
from threshold import Threshold
from sklearn.model_selection import GridSearchCV
from osvm import OrdinalSVM
from balanced_bag import BalancedBag
from balanced_smote import SMOTE, MSMOTE
from osvm_rank import RankOSVM
import sys

PAPER = 'iwann2017'
KERNEL = 'linear'
CROSS_VALIDATION = True
C_validation = np.logspace(-3, 3, 7)
NJOBS = -1
NPROCESSES = 1

# ensure there is a temp folder
if not os.path.exists('temp'):
    os.mkdir('temp')


def SVM_Linear(balanced, C, fit_intercept=True):
    balanced = 'balanced' if balanced else None
    return LinearSVC(
        class_weight=balanced, penalty='l1', tol=1e-3, dual=False, C=C,
        fit_intercept=fit_intercept)


def SVM_RBF(balanced, C):
    balanced = 'balanced' if balanced else None
    return SVC(class_weight=balanced, C=C)


if KERNEL == 'linear':
    models = [
        ('linear svm', SVM_Linear(False, 1), 'C'),
        ('linear svm weights', SVM_Linear(True, 1), 'C'),
        #('linear ranksvm ltr', Threshold(RankSVM(SVM_Linear(False, 1)), 'ltr'),
        # 'model__model__C'),
        #('linear ranksvm two', Threshold(RankSVM(SVM_Linear(False, 1)), 'between'),
        # 'model__model__C'),
        #('linear ranksvm unif', Threshold(RankSVM(SVM_Linear(False, 1)),
        # 'uniform'), 'model__model__C'),
        ('linear ranksvm inv', Threshold(RankSVM(SVM_Linear(False, 1)), 'inverse'),
         'model__model__C'),
        ('linear ranksvm abs',
         Threshold(RankSVM(SVM_Linear(False, 1)), 'absolute'),
         'model__model__C'),
        #('linear svmlight ltr', Threshold(SVMLight(1, 'linear'), 'ltr'), 'model__C'),
        ('linear svorim', SVORIM(1, 'linear'), 'C'),
        ('linear svorex', SVOREX(1, 'linear'), 'C'),
        ('linear osvm', OrdinalSVM(SVM_Linear(False, 1, False)), 'model__C'),
        ## NEW: iwann2017 work
        ('linear bag ranksvm inv',
         BalancedBag(Threshold(RankSVM(SVM_Linear(False, 1)), 'inverse'), 10,
                     'median'),
         'model__model__model__C'),
        ('linear bag ranksvm abs',
         BalancedBag(Threshold(RankSVM(SVM_Linear(False, 1)), 'absolute'), 10,
                     'median'),
         'model__model__model__C'),
        ('linear smote ranksvm inv',
         SMOTE(Threshold(RankSVM(SVM_Linear(False, 1)), 'inverse')),
         'model__model__model__C'),
        ('linear smote ranksvm abs',
         SMOTE(Threshold(RankSVM(SVM_Linear(False, 1)), 'absolute')),
         'model__model__model__C'),
        ('linear msmote ranksvm inv',
         MSMOTE(Threshold(RankSVM(SVM_Linear(False, 1)), 'inverse')),
         'model__model__model__C'),
        ('linear msmote ranksvm abs',
         MSMOTE(Threshold(RankSVM(SVM_Linear(False, 1)), 'absolute')),
         'model__model__model__C'),
        # TEST: try ordinal SVM using Ranking approach
        ('linear osvm rank abs', RankOSVM(SVM_Linear(False, 1), 'absolute'),
         'model__C'),
        ('linear osvm rank inv', RankOSVM(SVM_Linear(False, 1), 'inverse'),
         'model__C'),
    ]
    excluding = [
        #'nursery'  # ibpria2017
        'abalone5', 'abalone10', 'nursery'  # iwann2017
    ]
else:
    models = [
        ('rbf svm', SVM_RBF(False, 1), 'C'),
        ('rbf svm weights', SVM_RBF(True, 1), 'C'),
        ('rbf svmlight abs', Threshold(SVMLight(1, 'rbf'), 'absolute'), 'model__C'),
        #('rbf svmlight unif', Threshold(SVMLight(1, 'rbf'), 'uniform'), 'model__C'),
        ('rbf svmlight inv', Threshold(SVMLight(1, 'rbf'), 'inverse'), 'model__C'),
        ('rbf svorim', SVORIM(1, 'rbf'), 'C'),
        ('rbf svorex', SVOREX(1, 'rbf'), 'C'),
        #('rbf osvm', OrdinalSVM(SVM_RBF(False, 1)), 'model__C'),
        ## NEW: iwann2017 work
        ('rbf bag ranksvm inv',
         BalancedBag(Threshold(SVMLight(1, 'rbf'), 'inverse'), 10,
            'median'), 'model__model__C'),
        ('rbf bag ranksvm abs',
         BalancedBag(Threshold(SVMLight(1, 'rbf'), 'absolute'), 10,
            'median'), 'model__model__C'),
        ('rbf smote ranksvm inv',
         SMOTE(Threshold(SVMLight(1, 'rbf'), 'inverse')), 'model__model__C'),
        ('rbf smote ranksvm abs',
         SMOTE(Threshold(SVMLight(1, 'rbf'), 'absolute')), 'model__model__C'),
        ('rbf msmote ranksvm inv',
         MSMOTE(Threshold(SVMLight(1, 'rbf'), 'inverse')), 'model__model__C'),
        ('rbf msmote ranksvm abs',
         MSMOTE(Threshold(SVMLight(1, 'rbf'), 'absolute')), 'model__model__C'),
    ]
    excluding = [
        # ibpria2017
        'ERA', 'ERA1vs23456vs7vs8vs9', 'ESL', 'ESL12vs3vs456vs7vs89', 'LEV',
        'SWD', 'abalone10', 'abalone5', 'nursery',
        #'triazines10', 'triazines5', 'cooling', 'stock10', 'car'
        # lento
        #'balance-scale'
    ]

if PAPER == 'iwann2017':
    # filter inverse threshold models
    models = [(name, model, param_C) for name, model, param_C in models
              if 'inv' not in name]

score_fns = [
    ('mae', mean_absolute_error),
    ('mmae', maximum_mean_absolute_error),
    ('tau', lambda y, yp: kendalltau(y, yp)[0]),
]


def model_fit_predict(model, param, Xtr, ytr, Xts):
    params = np.logspace(-3, 0, 4)
    if CROSS_VALIDATION:
        # duplicate classes too few: avoids errors in svorim and so on
        if np.any(np.bincount(ytr) < 4):
            for k in np.where(np.bincount(ytr) < 4)[0]:
                x = Xtr[ytr == k]
                for _ in range(4):
                    Xtr = np.r_[Xtr, x]
                    ytr = np.r_[ytr, np.repeat(k, len(x))]

        m = GridSearchCV(
            model, {param: params}, 'neg_mean_absolute_error', n_jobs=NJOBS)
        m.fit(Xtr, ytr)
        m = m.best_estimator_
    else:
        m = model.fit(Xtr, ytr)
    return m

print('kernel? ' + KERNEL)
print('cross validation? ' + ('yes' if CROSS_VALIDATION else 'no'))
print()
print(' '*(30+1) + ' '.join(['%-6s' % name for name, _ in score_fns]))
print()

dirname = os.path.join('../data', '30HoldoutOrdinalImbalancedDatasets')
datasets = sorted(os.listdir(dirname))
datasets = [dataset for dataset in datasets if dataset not in excluding]


def process_data(data):
    dirname2 = os.path.join(dirname, data, 'matlab')
    folds = len(os.listdir(dirname2)) // 2
    scores = np.zeros((len(models), len(score_fns)))
    scores_pred = np.zeros(len(models))

    for fold in range(folds):
        tr_filename = os.path.join(dirname2, 'train_%s.%d' % (data, fold))
        ts_filename = os.path.join(dirname2, 'test_%s.%d' % (data, fold))

        tr = np.loadtxt(tr_filename)
        Xtr = tr[:, :-1]
        ytr = tr[:, -1].astype(int)
        ts = np.loadtxt(ts_filename)
        Xts = ts[:, :-1]
        yts = ts[:, -1].astype(int)

        # start in zero: [0,K[
        ytr -= 1
        yts -= 1

        #if np.any(np.bincount(ytr) < 3):
        #    break
        if NPROCESSES == 1 and fold == 0:
            print('== DATASET %s ==' % data)
            print(np.unique(ytr))

        for i, (name, model, param) in enumerate(models):
            progress = (i+len(models)*fold) / (folds*len(models))
            if NPROCESSES == 1:
                sys.stdout.write('\r%4.1f%% fold%02d %-30s' % (100*progress, fold, name))
                sys.stdout.flush()

            _name = name.replace(' ', '-')
            out_filename = '../out/%s--%s--%d--yp.csv' % (_name, data, fold)
            if os.path.exists(out_filename):
                continue

            model = model_fit_predict(model, param, Xtr, ytr, Xts)
            yp = model.predict(Xts)
            if CROSS_VALIDATION:  # only save when cross-validating
                np.savetxt(out_filename, yp, '%d')

            for j, (_, fn) in enumerate(score_fns):
                s = fn(yts, yp)
                scores[i, j] += s / folds
            if isinstance(model, Threshold):
                pp = model.predict_proba(Xts)
                scores_pred[i] += kendalltau(yts, pp)[0] / folds
                np.savetxt('../out/%s--%s--%d--pp.csv' % (_name, data, fold), pp)
    if NPROCESSES == 1:
        sys.stdout.write('\r                                 \r')

    if fold > 0:
        if NPROCESSES > 1:
            print('== DATASET %s ==' % data)
            print(np.unique(ytr))

        mins = np.nanmin(scores, 0)
        maxs = np.nanmax(scores, 0)
        for i, (name, _, _) in enumerate(models):
            def symb(s, ma, mi):
                return '+' if s == ma else '-' if s == mi else ' '
            s = ' '.join(['%5.3f' % s + symb(s, maxs[j], mins[j])
                          for j, s in enumerate(scores[i])])
            if scores_pred[i]:
                s += ' %.3f' % scores_pred[i]
            print('%-30s %s' % (name, s))
        print()


if NPROCESSES > 1:
    import multiprocessing
    pool = multiprocessing.Pool()
    pool.map(process_data, datasets)
else:
    for dataset in datasets:
        process_data(dataset)
