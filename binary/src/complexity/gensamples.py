# -*- coding: utf-8 -*-

# Generate shynthetic unbalanced cases.

import numpy as np
import samples

N0 = 500
N1 = 50
stddev = 0.25

s = (
    ('disjunct', lambda: samples.xor(N0=N0, N1=N1, stddev=0)),
    ('overlap', lambda: samples.circle2(N0=N0, N1=N1, stddev=stddev)),
    ('disjunct + overlap', lambda: samples.xor(N0=N0, N1=N1, stddev=stddev)),
    ('none', lambda: samples.circle2(N0=N0, N1=N1)),
)

import matplotlib.pyplot as plot
plot.clf()
for i, (name, fn) in enumerate(s):
    plot.subplot(2, 2, i+1)
    X, y = fn()
    plot.plot(X[y == 0, 0], X[y == 0, 1], '.', mew=1, ms=2, color='lightgray')
    plot.plot(X[y == 1, 0], X[y == 1, 1], 'x', mew=1, ms=4, color='black')

    n0 = np.sum(y == 0)
    n1 = len(y) - n0
    print 'sample %d - N0: %d, N1: %d, IR: %.3f' % (i, n0, n1, n1/float(n0))

    plot.title(name)
    plot.xticks([])
    plot.yticks([])
