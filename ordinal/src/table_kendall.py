from scipy.stats import kendalltau
import numpy as np
import os
import sys
from ttest_paired import ttest_paired

kernel = 'linear'

FOLDS = 30

datasets = os.listdir('../data/30HoldoutOrdinalImbalancedDatasets')
files = os.listdir('../out')

models = []
for file in files:
    if not file.startswith(kernel):
        continue
    model, dataset, fold, _ = file.split('--')
    if model not in models and ('rank' in model or 'light' in model):
        models.append(model)
models = sorted(models)

datasets = sorted(datasets, key=str.lower)

yp_tau = np.zeros((len(datasets), len(models)))
pp_tau = np.zeros((len(datasets), len(models)))

yp_folds = [[[] for _ in models] for _ in datasets]
pp_folds = [[[] for _ in models] for _ in datasets]


for j, dataset in enumerate(datasets):
    for fold in range(FOLDS):
        data_filename = '../data/30HoldoutOrdinalImbalancedDatasets/%s/matlab/test_%s.%d' % (dataset, dataset, fold)
        y = np.loadtxt(data_filename, int, usecols=[-1])
        y -= 1  # [0,k[
        for k, model in enumerate(models):
            progress = 100*(j*len(models)*FOLDS+fold*len(models)+k) / (len(datasets)*len(models)*FOLDS)
            sys.stdout.write('\r%-24s %.1f%%' % (dataset, progress))
            yp_filename = '../out/%s--%s--%d--yp.csv' % (model, dataset, fold)
            pp_filename = '../out/%s--%s--%d--pp.csv' % (model, dataset, fold)
            if not os.path.exists(yp_filename):
                print('Warning: %s fold %d for %s not generated yet' % (dataset, fold, model))
                yp_tau[j, k] = 0
                pp_tau[j, k] = 0
                continue
            yp = np.loadtxt(yp_filename).astype(int)
            pp = np.loadtxt(pp_filename).astype(int)

            yp_tau[j, k] += kendalltau(y, yp)[0] / FOLDS
            pp_tau[j, k] += kendalltau(y, pp)[0] / FOLDS

            yp_folds[j][k].append(kendalltau(y, yp)[0])
            pp_folds[j][k].append(kendalltau(y, pp)[0])
sys.stdout.write('\r                                                      \r')

# remove lines at zero
ix = np.where(np.logical_not(np.all(yp_tau == 0, 1)))[0]
print(np.mean(yp_tau[ix] > pp_tau[ix]))

for k, model in enumerate(models):
    yp_better = 0
    pp_better = 0
    magnitude = 0
    for j in ix:
        if ttest_paired(yp_folds[j][k], pp_folds[j][k]) < 0.05:
            pp_better += 1
        elif ttest_paired(pp_folds[j][k], yp_folds[j][k]) < 0.05:
            yp_better += 1
        magnitude += np.mean(yp_tau[ix, k]-pp_tau[ix, k])
    print(model)
    print('threshold better: %.2f, score better: %.2f' % (
        yp_better/len(ix), pp_better/len(ix)))
    print('threshold better by an average magnitude of %.2f' % (magnitude/len(ix)))
    print()
