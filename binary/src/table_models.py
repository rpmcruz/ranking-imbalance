#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import load_csv
import scipy.stats
import numpy as np
import itertools
import sys
import os

import warnings
warnings.filterwarnings('ignore')
# We will ignore sklearn warnings because of this sklearn bug:
# https://github.com/scikit-learn/scikit-learn/issues/2586
# It issues false warnings for F1-score when recall or prec=0.
# NOTE: we still show important errors and exceptions.


#from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


def gmean(y, yp):
    TP = np.sum(np.logical_and(y == 1, yp == 1))
    TN = np.sum(np.logical_and(y == 0, yp == 0))
    return np.sqrt(TP*TN)/len(y)

scores = (
    ('Prec', precision_score),
    ('Recall', recall_score),
    (r'$\mathit{F_1}$', f1_score),
    ('gmean', gmean),
    ('ROC AUC', roc_auc_score),
)


def ztest_onetailed(x1, x2):  # H0: x1 = x2, H1: x1 < x2
    mu = np.mean(x1) - np.mean(x2)
    stderr = np.sqrt(np.var(x1)/len(x1) + np.var(x2)/len(x2))
    if stderr == 0:
        return 1
    z = mu / stderr
    return scipy.stats.norm.cdf(z)


def ttest_paired(x1, x2):  # H0: x1 = x2, H1: x1 < x2
    x = np.asarray(x1) - np.asarray(x2)
    mu = np.mean(x)
    stderr = np.sqrt(np.var(x)/len(x))
    if stderr == 0:
        return 1
    t = mu / stderr
    df = len(x)-1
    return scipy.stats.t.cdf(t, df)


def write_model(models, filename, familyname, standalone):
    if len(models) == 6:
        columns = ('Ranking', 'Baseline', 'Weights', 'SMOTE', 'MSMOTE', 'MetaCost')
    else:
        print 'Warning: non-standard headers!'
        columns = models

    samples = sorted(os.listdir('../data'), key=lambda s: s.lower())
    nfolds = 40

    # Sort samples by imbalance ratio
    # descending sort by minority ratio

    ratios = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        X, y = load_csv(os.path.join('../data', sample))
        ratio = lambda X, y: np.sum(y)/float(len(y))
        ratios[i] = ratio(X, y)

    idxs = ratios.argsort()[::-1]
    samples = [samples[i] for i in idxs]

    # Build table

    tex = open('../out/%s.tex' % filename, 'w')

    if standalone:
        tex.write(r'''\documentclass{standalone}
\usepackage{graphicx}
% latex HACK so that bold font have same size as rest (for the columns)
\newsavebox\CBox
\def\textBF#1{\sbox\CBox{#1}\resizebox{\wd\CBox}{\ht\CBox}{\textbf{#1}}}
\begin{document}
    ''')
    tex.write(r'''\begin{tabular}{|l|''')
    for i in scores:
        tex.write('|')
        tex.write('p{3.8em}|' * len(columns))
    tex.write('}\\hline\n')

    tex.write('\\textit{%s}' % familyname)
    for scorei, (name, _) in enumerate(scores):
        border = '|' if scorei == len(scores)-1 else '||'
        tex.write(' & \\multicolumn{%d}{c%s}{\\textit{%s}}' % (len(columns),
                  border, name))
    tex.write('\\\\\n')

    tex.write('Sample')  # put sample name
    tex.write((' &' + '&'.join(columns)) * len(scores))
    tex.write('\\\\\\hline\n')

    ssmean = np.zeros((len(scores), len(models)))  # samples score mean
    tops = np.zeros((len(scores), len(models)), int)

    for samplei, sample in enumerate(samples):
        sample = sample[:-4]
        tex.write('%s' % sample)

        for scorei, (_, score_fn) in enumerate(scores):
            sys.stdout.write('\r%s: %2d%%' % (
                familyname, ((samplei*len(scores)+scorei)*100) /
                    (len(samples)*len(scores))))
            sys.stdout.flush()
            sfolds = [[] for _ in models]

            for modeli, model in enumerate(models):
                for fold in xrange(nfolds):
                    try:
                        ffold = '%s-fold%02d.csv' % (sample, fold+1)
                        y = np.loadtxt(os.path.join('../out/test', ffold),
                                       delimiter=',')[:, -1].astype(int)
                        yp = np.loadtxt(os.path.join('../out/predictions', model,
                                                     ffold), int)
                        ys = np.loadtxt(os.path.join('../out/scores', model,
                                                     ffold))
                        if len(y) != len(yp) or len(y) != len(ys):
                            print 'Error: different predictions length for ' \
                                '%-21s %s' % (sample, model)
                    except:
                        pass
                    else:
                        if score_fn == roc_auc_score:
                            if len(ys.shape) > 1:
                                ys = ys[:, 1]
                            s = roc_auc_score(y, ys)
                        else:
                            s = score_fn(y, yp)
                        sfolds[modeli].append(s)
                if len(sfolds[modeli]) < nfolds and len(sfolds[modeli]) > 0:
                    print 'Warning: only %2d folds exist for %-21s %s' % (
                        len(sfolds[modeli]), sample, model)

            smean = [np.mean(sfolds[i]) for i in xrange(len(models))]
            ssmean[scorei] += smean
            sstd = [np.std(sfolds[i], ddof=1) for i in xrange(len(models))]

            best_model = np.argmax(smean)
            if np.isnan(smean[best_model]):
                best_model = -1

            best = np.zeros(len(models), bool)
            if best_model >= 0:
                best[best_model] = True
                for modeli in xrange(len(models)):
                    x1 = sfolds[modeli]
                    x2 = sfolds[best_model]
                    if modeli != best_model and len(x1) == len(x2) and len(x1) > 0:
                        # vou tentar rejeitar H0: F1(modeli) = F1(best_model)
                        # de forma a mostrar que H1: F1(modeli) < F1(best_model)
                        #pvalue = ztest_onetailed(x1, x2)
                        pvalue = ttest_paired(x1, x2)  # emparelhado!
                        same = not (pvalue < 0.05)  # cannot reject
                        """
                        if round(smean[best_model], 2) == round(smean[modeli], 2):
                            same = True  # HACK but necessary
                        """
                        best[modeli] = same
                        if False:
                            print 'H0: %f +- %f = %f +- %f -> pvalue=%f ? %d' \
                                % (smean[modeli], sstd[modeli], smean[best_model],
                                   sstd[best_model], pvalue, best[modeli])
            tops[scorei] += best

            for modeli in xrange(len(models)):
                bold = best[modeli]
                if len(sfolds[modeli]) == 0:
                    str = '--'
                    bold = False
                else:
                    #str = '%.2f $\pm$ %.2f' % (smean[modeli], sstd[modeli])
                    str = '%.3f' % smean[modeli]
                if bold:
                    tex.write('& \\textBF{%s}' % str)
                else:
                    tex.write('& %s' % str)

        tex.write('\\\\\n')
    sys.stdout.write('\r                            \r')

    tex.write('\\hline\n')
    tex.write(r'Average')
    for scorei in xrange(len(scores)):
        for modeli in xrange(len(models)):
            s = ssmean[scorei, modeli]
            tex.write(' & %.3f' % (s/len(samples)))
    tex.write('\\\\\n')
    tex.write(r'Winner')
    for scorei in xrange(len(scores)):
        for modeli in xrange(len(models)):
            s = tops[scorei, modeli]
            tex.write(' & %d\\%%' % (100*float(s)/len(samples)))

    #tex.write('\\\\\\hline\n')
    #for model in models:  # print model names again for readability
    #    tex.write(' & %s' % model)
    tex.write(r'''\\\hline
\end{tabular}
    ''')
    if standalone:
        tex.write(r'''\end{document}
''')
    tex.close()


if __name__ == '__main__':
    #models = sorted(os.listdir('../out/predictions'), key=lambda s: s.lower())

    models = (
        ('RankSVM', 'SVM Linear', 'SVM Linear b', 'SVM Linear SMOTE',
         'SVM Linear MSMOTE', 'SVM Linear MetaCost'),
        ('RankBoost', 'AdaBoost', 'AdaBoost b', 'AdaBoost SMOTE',
         'AdaBoost MSMOTE', 'AdaBoost MetaCost'),
        ('RankNet', 'NeuralNet', 'NeuralNet b', 'NeuralNet SMOTE',
         'NeuralNet MSMOTE', 'NeuralNet MetaCost'),
        ('GBRank', 'Random Forest', 'Random Forest b', 'Random Forest SMOTE',
         'Random Forest MSMOTE', 'Random Forest MetaCost'),
    )
    filenames = (
        'linear-svm',
        'adaboost',
        'nnet',
        'trees',
    )
    familynames = (
        'Linear SVM',
        'AdaBoost',
        'Neural Networks',
        'Decision Trees',
    )

    for prefix, standalone in (('standalone-', True), ('', False)):
        for model, filename, familyname in itertools.izip(
                models, filenames, familynames):
            write_model(model, prefix+filename, familyname, standalone)
