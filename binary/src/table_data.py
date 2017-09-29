#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Metrics table of the empirical data for publishing.

import os
import sys
import itertools
import utils
import numpy as np
from complexity import metrics

scores = (
    ('N', lambda X, y: X.shape[0], 0),
    ('Features', lambda X, y: X.shape[1], 0),
    ('IR', lambda X, y: np.sum(y)/float(len(y)), 3),
    ('Overlap', metrics.overlap1, 3),
    #('Disjunct', metrics.silhouette_positive, 3),
)

standalone = False

samples = os.listdir('../data')

# gen table

table = np.zeros((len(samples), len(scores)))

for i, sample in enumerate(samples):
    sys.stdout.write('\r%3d%%' % ((i*100)/len(samples)))
    sys.stdout.flush()
    for j, (_, fn, _) in enumerate(scores):
        X, y = utils.load_csv(os.path.join('../data', sample))
        table[i, j] = fn(X, y)
sys.stdout.write('\rsaving...\r')

# descending sort by minority ratio

idxs = table[:, 2].argsort()[::-1]
table = table[idxs]
samples = [samples[i] for i in idxs]

# export table

tex = open('../out/data.tex', 'w')

if standalone:
    tex.write(r'''\documentclass{standalone}
\begin{document}
''')
tex.write(r'\begin{tabular}{|l|l')
for i in xrange(len(scores)):
    if i == 3:
        tex.write(r'|r')  # separate our metrics
    else:
        tex.write(r'|r')
tex.write('|}\\hline\nDataset & Minority')
for i, (name, _, _) in enumerate(scores):
    if i == 2:  # bold on ratio where we order
        tex.write(' & \\textbf{%s}' % name)
    else:
        tex.write(' & %s' % name)
tex.write('\\\\\\hline\n')

for filename, row in itertools.izip(samples, table):
    # latex friendly
    filename = filename[:-4].replace(r'_', r'\_')

    # regex for this would be a little complicated because we have multiply
    # slashes in the filenames -- this way is faster anyhow
    i = filename.rfind('-')
    if i == -1:
        name = filename
        attrb = '---'
    else:
        name = name = filename[0:i]
        attrb = filename[(i+1):]
    tex.write('%s & %s ' % (name, attrb))

    # attributes
    for i, nbr in enumerate(row):
        tex.write(('& %%.%df' % scores[i][2]) % nbr)
    tex.write('\\\\\n')
tex.write('\\hline\n')

tex.write(r'\end{tabular}')
if standalone:
    tex.write('\n\\end{document}')
tex.close()
sys.stdout.write('\r                \r')
