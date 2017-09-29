from scores import maximum_mean_absolute_error
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
import sys
from ttest_paired import ttest_paired

PAPER = 'iwann2017'


def natural_sort(l):
    # https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    import re
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

for kernel in ['linear', 'rbf']:
    score_fns = [
        ('mae', 'MAE', mean_absolute_error, -1),
        ('mmae', 'MMAE', maximum_mean_absolute_error, -1),
        ('tau', 'Corr', lambda y, yp: kendalltau(y, yp)[0], +1),
    ]
    if PAPER == 'iwann2017':
        score_fns = score_fns[:-1]

    FOLDS = 30

    datasets = os.listdir('../data/30HoldoutOrdinalImbalancedDatasets')
    files = os.listdir('../out')

    models = []
    for file in files:
        if not file.startswith(kernel):
            continue
        model, dataset, fold, _ = file.split('--')
        if model not in models:
            models.append(model)

    if PAPER == 'ibpria2017':
        MODELS = [
            ['ranksvm-abs', 'Rank Abs', 2.6],
            ['ranksvm-inv', 'Rank Inv', 2.6],
            ['svm', 'OvR SVM', 2.3],
            ['svm-weights', 'OvR SVM/w', 3.4],
            ['svorex', 'SVOREX', 4.0],
            ['svorim', 'SVORIM', 4.0],
            ['osvm', 'oSVM', 2.8],
        ]
        if kernel == 'rbf':
            MODELS[0][0] = 'svmlight-abs'
            MODELS[0][2] += 0.3
            MODELS[1][0] = 'svmlight-inv'
            MODELS[1][2] += 0.3
    elif PAPER == 'iwann2017':
        MODELS = [
            ['ranksvm-abs', r'WRank', 2.5],
            ['bag-ranksvm-abs', r'BRank', 2.5],
            ['smote-ranksvm-abs', 'SRank', 2.5],
            ['msmote-ranksvm-abs', 'MSRank', 2.5],
            ['svm', 'OvR', 2.5],
            ['svm-weights', 'OvR/w', 2.5],
            ['svorex', 'SVOREX', 4.0],
            ['svorim', 'SVORIM', 4.0],
            ['osvm', 'oSVM', 2.8],
            #['osvm-rank-abs', 'oSVM Rank Abs', 3.6],
            #['osvm-rank-abs', 'oSVM Rank Abs', 3.6],
        ]
        if kernel == 'rbf':
            MODELS[0][0] = 'svmlight-abs'
            #MODELS[1][0] = 'svmlight-inv'

    # re-order and exclude unwanted models
    _models = models[:]
    models = []
    for model, _, _ in MODELS:
        for m in _models:
            if m == kernel + '-' + model:
                models.append(m)
                break
    MODELS = [(name, latex, size) for name, latex, size in MODELS
              if kernel + '-' + name in models]

    if PAPER == 'iwann2017':
        excluding = [
            # ibpria2017
            'ERA', 'ERA1vs23456vs7vs8vs9', 'ESL', 'ESL12vs3vs456vs7vs89', 'LEV',
            'SWD', 'abalone10', 'abalone5', 'nursery',
            #'triazines10', 'triazines5', 'cooling', 'stock10', 'car'
            # lento
            #'balance-scale'
        ]
        datasets = [dataset for dataset in datasets if dataset not in excluding]

    natural_sort(datasets)
    scores = np.zeros((len(score_fns), len(datasets), len(models)))
    sfolds = [[[[] for _ in models] for _ in datasets] for _ in score_fns]

    for j, dataset in enumerate(datasets):
        for fold in range(FOLDS):
            data_filename = '../data/30HoldoutOrdinalImbalancedDatasets/%s/matlab/test_%s.%d' % (dataset, dataset, fold)
            y = np.loadtxt(data_filename, int, usecols=[-1])
            y -= 1  # [0,k[
            for k, model in enumerate(models):
                progress = 100*(j*len(models)*FOLDS+fold*len(models)+k) / (len(datasets)*len(models)*FOLDS)
                sys.stdout.write('\r%4.1f%% %-24s' % (progress, dataset))
                pred_filename = '../out/%s--%s--%d--yp.csv' % (model, dataset, fold)
                if not os.path.exists(pred_filename):
                    print('Warning: %s fold %d for %s not generated yet' % (dataset, fold, model))
                    scores[:, j, k] = 0
                    continue
                yp = np.loadtxt(pred_filename).astype(int)
                for i, (_, _, s, _) in enumerate(score_fns):
                    _s = s(y, yp)
                    scores[i, j, k] += _s / FOLDS
                    sfolds[i][j][k].append(_s)
    sys.stdout.write('\r                                                      \r')

    # replace nans by zero
    scores = np.nan_to_num(scores)

    if PAPER == 'iwann2017':
        f = open('../doc/iwann2017/tables/%s.tex' % kernel, 'w')
        f.write(r'''\documentclass{standalone}
\begin{document}
\begin{tabular}{|p{11em}''')
        for model, latex, size in MODELS:
            f.write('|l')
        f.write('|}\hline\n')

    for i, (score_name, score_tex_name, _, best) in enumerate(score_fns):
        if PAPER == 'ibpria2017':
            f = open('../doc/ibpria2017/tables/%s-%s.tex' % (kernel, score_name), 'w')
            f.write(r'''\documentclass{standalone}
\begin{document}
\begin{tabular}{|p{11em}''')
            for model, latex, size in MODELS:
                f.write('|p{%.1fem}' % size)
            f.write('|}\hline\n')

        if PAPER == 'iwann2017':
            f.write('\\multicolumn{%d}{|c|}{%s}\\\\\hline\n' % (len(MODELS)+1, score_tex_name))

        tops = np.zeros((len(datasets), len(models)), int)

        if best == -1:
            sup, argsup = np.nanmin, np.nanargmin
        else:
            sup, argsup = np.nanmax, np.nanargmax

        names = []
        for model in models:
            for m, latex, _ in MODELS:
                if model == kernel + '-' + m:
                    if 'Rank' in latex:
                        latex = r'\textbf{' + latex + '}'
                    names.append(latex)
        #first_column = score_tex_name
        f.write('Dataset & ' + ('& '.join(names)) + r'\\\hline' + '\n')

        for j, dataset in enumerate(datasets):
            if np.all(scores[:, j, :] == 0):
                # not generated yet
                continue
            #f.write('%02d' % (j+1))
            f.write(dataset)

            best_model = argsup(scores[i, j, :])
            for k, _ in enumerate(models):
                s = scores[i, j, k]
                if s != 0:
                    if best_model == k:
                        bold = True
                    else:
                        pvalue = ttest_paired(
                            sfolds[i][j][k], sfolds[i][j][best_model])
                        if np.isnan(pvalue):
                            bold = False
                        else:
                            if best == -1:
                                bold = not (pvalue > 0.95)  # cannot reject
                            else:
                                bold = not (pvalue < 0.05)  # cannot reject
                    if bold:
                        f.write('& \\bf %.2f' % s)
                        tops[j, k] += 1
                    elif np.isnan(s):
                        f.write('& --')
                    else:
                        f.write('& %.2f' % s)
                else:
                    f.write('&')
            f.write('\\\\\n')
        f.write('\\hline\n')

        f.write('Average & ' + '&'.join(['%.2f' % s for s in np.nanmean(scores[i], 0)]) + '\\\\\n')

        f.write('Deviation & ' + '&'.join(['%.2f' % s for s in np.nanstd(scores[i], 0)]) + '\\\\\n')

        datasets_total = np.sum(np.all(scores[i] != 0, 1))
        total = np.sum(tops, 0) / datasets_total
        total = [t if np.isfinite(t) else 0 for t in total]  # should not be needed
        f.write('Winner & ' + '&'.join(['%d\\%%' % (100*t) for t in total]) + '\\\\\hline\n')
        if PAPER == 'ibpria2017':
            f.write(r'''\end{tabular}
\end{document}''')
            f.close()
    if PAPER == 'iwann2017':
        f.write(r'''\end{tabular}
\end{document}''')
        f.close()
