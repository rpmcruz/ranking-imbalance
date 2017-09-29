#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import load_csv
from complexity import metrics
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import pickle
import sys
import os

nfolds = 40
families = (
    ('RankSVM', 'SVM Linear', 'SVM Linear b', 'SVM Linear SMOTE',
     'SVM Linear MSMOTE', 'SVM Linear MetaCost'),
    ('RankBoost', 'AdaBoost', 'AdaBoost b', 'AdaBoost SMOTE',
     'AdaBoost MSMOTE', 'AdaBoost MetaCost'),
    ('RankNet', 'NeuralNet', 'NeuralNet b', 'NeuralNet SMOTE',
     'NeuralNet MSMOTE', 'NeuralNet MetaCost'),
    #('GBRank', 'Random Forest', 'Random Forest b'),
)
family_names = ('Linear SVM', 'AdaBoost', 'NNet')

## Collect raw data
# Cache it in corr.pickle for fast retrieval

if os.path.exists('corr.pickle'):
    with open('corr.pickle') as f:
        folds, sfolds, afolds, safolds, IR, OR, sIR, sOR = pickle.load(f)
else:
    samples = os.listdir('../data')
    folds = [[[] for _ in models] for models in families]
    sfolds = [[np.zeros(len(samples)) for _ in models] for models in families]
    afolds = [[[] for _ in models] for models in families]
    safolds = [[np.zeros(len(samples)) for _ in models] for models in families]
    IR = []
    OR = []
    sIR = []
    sOR = []

    for samplei, sample in enumerate(samples):
        X, y = load_csv(os.path.join('../data', sample))
        ratio = lambda X, y: np.sum(y)/float(len(y))
        sIR.append(ratio(X, y))
        sOR.append(metrics.overlap1(X, y))

        sample = sample[:-4]
        for fold in xrange(nfolds):
            sys.stdout.write('\r%2d%%' % (
                ((samplei*nfolds + fold)*100) /
                (len(samples)*nfolds)))
            sys.stdout.flush()

            ffold = '%s-fold%02d.csv' % (sample, fold+1)
            X = np.loadtxt(os.path.join('../out/test', ffold), delimiter=',')
            y = X[:, -1].astype(int)
            X = X[:, 0:-1]

            IR.append(ratio(X, y))
            OR.append(metrics.overlap1(X, y))

            for familyi, models in enumerate(families):
                for modeli, model in enumerate(models):
                    try:
                        yp = np.loadtxt(os.path.join('../out/predictions', model,
                                                     ffold), int)
                        ys = np.loadtxt(os.path.join('../out/scores', model,
                                                     ffold))
                        if len(y) != len(yp) or len(y) != len(ys):
                            print 'Error: different predictions length for ' \
                                '%-21s %s' % (sample, model)
                    except:
                        print "Warning: not found (%d, %s) predictions/scores" % (
                            familyi, model)
                    else:
                        s = f1_score(y, yp)
                        folds[familyi][modeli].append(s)
                        sfolds[familyi][modeli][samplei] += s / nfolds
                        if len(ys.shape) > 1:
                            ys = ys[:, 1]
                        s = roc_auc_score(y, ys)
                        afolds[familyi][modeli].append(s)
                        safolds[familyi][modeli][samplei] += s / nfolds
    sys.stdout.write('\r                \r')

    IR = np.asarray(IR)
    OR = np.asarray(OR)
    sIR = np.asarray(sIR)
    sOR = np.asarray(sOR)
    for familyi in xrange(len(families)):
        for modeli in xrange(len(models)):
            folds[familyi][modeli] = np.asarray(folds[familyi][modeli])
            afolds[familyi][modeli] = np.asarray(afolds[familyi][modeli])

    with open('corr.pickle', 'w') as f:
        pickle.dump([folds, sfolds, afolds, safolds, IR, OR, sIR, sOR], f)

## ANALYSIS

import scipy.stats

# intra-correlation (ranker versus others of the same family)

print '** Intra-correlations'
for familyi in xrange(len(families)):
    for modeli in xrange(len(families[familyi])):
        rank = folds[familyi][0]
        other = folds[familyi][modeli]
        res = scipy.stats.spearmanr(rank, other)[0]
        print '%-12s vs %-19s = %f' % (families[familyi][0],
                                       families[familyi][modeli], res)
    print

# inter-correlation (rankers of different families)

print '** Inter-correlations'
for familyi in xrange(len(families)):
    for familyj in xrange(familyi+1, len(families)):
        rank1 = folds[familyi][0]
        rank2 = folds[familyj][0]
        res = scipy.stats.spearmanr(rank1, rank2)[0]
        print '%-12s vs %-19s = %f' % (families[familyi][0],
                                       families[familyj][0], res)
print

from scipy.stats import spearmanr

def spearman_partial(x, y, z):
    # https://pt.wikipedia.org/wiki/Correla%C3%A7%C3%A3o_parcial
    # i.e. correlaçao entre x e y controlando para z
    cor_xy = spearmanr(x, y)[0]
    cor_xz = spearmanr(x, z)[0]
    cor_yz = spearmanr(y, z)[0]
    return (cor_xy - cor_xz*cor_yz) / \
        (np.sqrt(1-cor_xz**2)*np.sqrt(1-cor_yz**2))


# metrics correlations

print '** Metrics'
for metric, R in (('IR', IR), ('OR', OR)):
    for familyi in xrange(len(families)):
        for modeli, model in enumerate(folds[familyi]):
            rank = model
            res = scipy.stats.spearmanr(rank, R)[0]
            print '%-12s vs %-2s = %f' % (families[familyi][modeli], metric, res)
    print

print
print '** Metrics difference'
for familyi in xrange(len(families)):
    for modeli, model in enumerate(folds[familyi]):
        rank = model
        res = np.abs(scipy.stats.spearmanr(rank, OR)[0]) - \
            np.abs(scipy.stats.spearmanr(rank, IR)[0])
        print '%-12s vs OR-IR = %f' % (families[familyi][modeli], res)
print

print '** Metrics controlled'
for familyi in xrange(len(families)):
    for modeli, model in enumerate(folds[familyi]):
        rank = model
        res = spearman_partial(rank, IR, OR)
        print '%-12s vs IR/OR = %f' % (families[familyi][modeli], res)
print


latex = open('../out/corr.tex', 'w')
latex.write('\\begin{tabular}{|l|%s}\n' % ('l|' * len(families[0])))
latex.write('\\hline\n')
latex.write('Spearman\'s $\\rho$ & ')
latex.write(' &'.join(('Ranking', 'Baseline', 'Weights', 'SMOTE', 'MSMOTE', 'MetaCost')))
latex.write('\\\\\\hline\n')
for familyi in xrange(len(families)):
    latex.write('\\textit{%s}' % family_names[familyi])
    latex.write(' &' * 6)
    latex.write('\\\\\n')
    for metric, R in (('IR', IR), ('Overlap', OR)):
        latex.write(metric)
        res = np.zeros(6)
        for modeli, model in enumerate(folds[familyi]):
            rank = model
            res[modeli] = scipy.stats.spearmanr(rank, R)[0]
        _min = np.argmin(np.abs(res))
        _max = np.argmax(np.abs(res))
        for modeli, model in enumerate(folds[familyi]):
            _res = res[modeli]
            latex.write('& %.3f' % _res)
        latex.write('\\\\\n')
latex.write('\\hline\n')
latex.write(r'\end{tabular}')
latex.close()


# same, but using linear regression to control for confounder IR (still
# testing)

print '** Controlled Inter-correlations !'
for familyi in xrange(len(families)):
    for familyj in xrange(familyi+1, len(families)):
        rank1 = folds[familyi][0]
        rank2 = folds[familyj][0]

        res = spearman_partial(rank1, rank2, IR)
        print '%-12s vs %-19s = %f' % (families[familyi][0],
                                       families[familyj][0], res)


print '** Relative gains in scores'
for familyi in xrange(len(families)):
    res = sfolds[familyi][0] - np.amax(sfolds[familyi][1:6], 0)
    print 'F1  %-12s, avg=%.3f avg pos=%.3f avg neg=%.3f' % (
        family_names[familyi], np.mean(res), np.mean(res[res > 0]),
        np.mean(res[res < 0]))
    res = safolds[familyi][0] - np.amax(safolds[familyi][1:6], 0)
    print 'ROC %-12s, avg=%.3f avg pos=%.3f avg neg=%.3f' % (
        family_names[familyi], np.mean(res), np.mean(res[res > 0]),
        np.mean(res[res < 0]))



# inter table
latex = open('../out/corr-inter.tex', 'w')
latex.write('\\begin{tabular}{|%s}\n' % ('l|' * len(families[0])))
latex.write('\\hline\n')
latex.write('Spearman\'s $\\rho$ & ')
latex.write(' &'.join(('Baseline', 'Weights', 'SMOTE', 'MSMOTE', 'MetaCost')))
latex.write('\\\\\\hline\n')
for familyi in xrange(len(families)):
    latex.write(families[familyi][0])
    modeli = 0
    res = np.zeros(5)
    for modelj in xrange(1, len(families[familyi])):
        a = folds[familyi][modeli]
        b = folds[familyi][modelj]
        c = IR
        res[modelj-1] = spearman_partial(a, b, c)
    _max = np.floor(np.amax(np.abs(res))*100)/100
    _min = np.ceil(np.amin(np.abs(res))*100)/100
    for modelj in xrange(1, len(families[familyi])):
        _res = res[modelj-1]
        latex.write('& %.3f' % _res)
    latex.write('\\\\\n')
latex.write('\\hline\n')
latex.write(r'\end{tabular}')
latex.close()


# intra table
latex = open('../out/corr-intra.tex', 'w')
latex.write('\\begin{tabular}{|%s}\n' % ('l|l|' * len(families)))
latex.write('\\hline\n')
latex.write('Spearman\'s $\\rho$ ')
for family in families:
    latex.write(' & %s' % family[0])
latex.write('\\\\\\hline\n')
for familyi, family in enumerate(families):
    latex.write(family[0])
    res = np.zeros(len(families))
    for familyj in xrange(len(families)):
        a = folds[familyi][0]
        b = folds[familyj][0]
        c = IR
        res[familyj] = spearman_partial(a, b, c)
    _max = np.floor(np.amax(np.abs(res))*100)/100
    _min = np.ceil(np.amin(np.abs(res))*100)/100
    for _res in res:
        latex.write('& %.3f' % _res)
    latex.write('\\\\\n')
latex.write('\\hline\n')
latex.write(r'\end{tabular}')
latex.close()



## PLOTS

import matplotlib.pyplot as plot
plot.ioff()

for ymetric, _folds in (('F1', sfolds), ('ROC AUC', safolds)):
    for xmetric, R in (('IR', sIR), ('OR', sOR)):
        print '\n', xmetric
        i = np.argsort(R)
        x = R[i]
        for familyi, models in enumerate(families):
            for modeli, model in enumerate(models):
                y = _folds[familyi][modeli][i]
                plot.plot(x, y, label=model)
            plot.title(family_names[familyi])
            plot.xlabel(xmetric)
            plot.ylabel(ymetric)
            plot.ylim(0, 1)
            plot.legend(loc='lower right')
            plot.show()
