#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plot
import numpy as np
import itertools
import os

# we sort at this point and so avoid doing any sorting afterwards
files = sorted(os.listdir('../out/imbalances'))

labels = []
xs = []  # will be a list of models of a list of scores
ys = []

for file in files:
    if file.count('-') == 0 or file.count('.') == 0:
        print 'Warning: malformed file name: %s' % file
        continue
    s = np.loadtxt(os.path.join('../out/imbalances', file))
    model = file[0:file.find('-')]
    if model not in labels:
        modeli = len(labels)
        labels.append(model)
        x = []
        y = []
        xs.append(x)
        ys.append(y)
    else:
        modeli = labels.index(model)
        x = xs[modeli]
        y = ys[modeli]
    x.append(file[(file.find('-')+1):file.rfind('.')])
    y.append(s)

plot.ioff()
for x, y, label in itertools.izip(xs, ys, labels):
    plot.plot(x, y, '-o', label=label)
plot.legend(loc='upper left')
plot.show()
