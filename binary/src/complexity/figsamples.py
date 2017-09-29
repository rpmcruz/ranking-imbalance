# -*- coding: utf-8 -*-

# Generate figures for the shynthetic unbalanced cases.

import numpy as np
import samples
from export import latex_style

latex_style(1, 0.25)

N0 = 250
N1 = 24
stddev = 0.20

s = (
    ('disjunct', lambda: samples.xor(N0=N0, N1=N1, stddev=0)),
    ('overlap', lambda: samples.circle2(N0=N0, N1=N1, stddev=stddev)),
    ('disjunct + overlap', lambda: samples.xor(N0=N0, N1=N1, stddev=stddev)),
    ('none', lambda: samples.circle2(N0=N0, N1=N1)),
)

import matplotlib.pyplot as plot
plot.ioff()
for i, (name, fn) in enumerate(s):
    plot.clf()
    X, y = fn()
    plot.plot(X[y == 0, 0], X[y == 0, 1], '.', mew=1, ms=1, color='lightgray')
    plot.plot(X[y == 1, 0], X[y == 1, 1], 'x', mew=0.6, ms=2.5, color='black')

    n0 = np.sum(y == 0)
    n1 = len(y) - n0
    print 'sample %d - N0: %d, N1: %d, IR: %.3f' % (i, n0, n1, n1/float(n0))

    #plot.title(name)
    plot.xticks([])
    plot.yticks([])
    plot.tight_layout(0.08)
    plot.savefig('../../doc/sample-%d.png' % i)
    plot.show()
