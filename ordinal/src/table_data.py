#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Metrics table of the empirical data for publishing.

import os
import sys
import numpy as np

PAPER = 'iwann2017'


def natural_sort(l):
    # https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    import re
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

def min_nonzero(y):
    y = y.copy()
    y[y == 0] = 9999
    return np.amin(y)

scores = (
    ('N', lambda X, y: X.shape[0], 0),
    ('\#vars', lambda X, y: X.shape[1], 0),
    ('K', lambda X, y: len(np.unique(y)), 0),
    ('IR', lambda X, y: min_nonzero(np.bincount(y))/np.max(np.bincount(y)), 3),
    #('Overlap', metrics.overlap1, 3),
    #('Disjunct', metrics.silhouette_positive, 3),
)

samples = os.listdir('../data/30HoldoutOrdinalImbalancedDatasets')
natural_sort(samples)

if PAPER == 'iwann2017':
    excluding = [
        'ERA', 'ERA1vs23456vs7vs8vs9', 'ESL', 'ESL12vs3vs456vs7vs89', 'LEV',
        'SWD', 'abalone10', 'abalone5', 'nursery',
    ]
    samples = [s for s in samples if s not in excluding]

# gen table

table = np.zeros((len(samples), len(scores)))

for i, sample in enumerate(samples):
    sys.stdout.write('\r%3d%%' % ((i*100)/len(samples)))
    sys.stdout.flush()
    for j, (_, fn, _) in enumerate(scores):
        dirname = '../data/30HoldoutOrdinalImbalancedDatasets/%s/matlab/' % sample
        tr = np.loadtxt(dirname + '/train_%s.0' % sample)
        ts = np.loadtxt(dirname + '/test_%s.0' % sample)
        Xtr = tr[:, :-1]
        Xts = ts[:, :-1]
        ytr = tr[:, -1]
        yts = ts[:, -1]
        X = np.r_[Xtr, Xts]
        y = np.r_[ytr, yts].astype(int)
        table[i, j] = fn(X, y)
sys.stdout.write('\rsaving...\r')

# descending sort by minority ratio

#idxs = table[:, 2].argsort()[::-1]
#table = table[idxs]
#samples = [samples[i] for i in idxs]

# export table

tex = open('../doc/%s/tables/data.tex' % PAPER, 'w')

tex.write(r'''\documentclass{standalone}
\begin{document}
''')
tex.write(r'\begin{tabular}{|l')
for i in range(len(scores)):
    tex.write(r'|r')
#tex.write('|}\\hline\n\# & Dataset')
tex.write('|}\\hline\nDataset')
for i, (name, _, _) in enumerate(scores):
    tex.write(' & %s' % name)
tex.write('\\\\\\hline\n')

for i, (filename, row) in enumerate(zip(samples, table)):
    #tex.write('%02d & %s' % (start+i+1, filename))
    tex.write(filename)

    # attributes
    for i, nbr in enumerate(row):
        tex.write(('& %%.%df' % scores[i][2]) % nbr)
    tex.write('\\\\\n')
tex.write('\\hline\n')

tex.write(r'\end{tabular}')
tex.write('\n\\end{document}')
tex.close()
sys.stdout.write('\r                \r')
