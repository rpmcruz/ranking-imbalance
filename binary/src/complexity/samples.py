# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats

# A few synthetic and real samples.

# see https://developers.lyst.com/2015/05/08/iclr-2015/


def load(filename):
    if filename.endswith('.csv'):
        return load_csv('../test/' + filename)
    else:
        return globals()[filename](1000)


def load_csv(filename):
    with open(filename, 'r') as f:
        counts = [f.readline().count(x) for x in (' ', ',')]
        delimiter = max(zip((' ', ','), counts), key=lambda x: x[1])[0]
    data = np.loadtxt(filename, delimiter=delimiter)
    X = data[:, 0:-1]
    y = data[:, -1]
    import collections
    print 'loading data %-35s: %s' % (filename, collections.Counter(y))
    return (X, y)


def circle(n=1000, r1=4, r2=2, offset=1):
    assert r1 > r2
    X = (np.random.rand(n, 2)*2-1)*r1
    y = (X[:, 0]-offset)**2+(X[:, 1]-offset)**2 <= r2**2
    X += np.random.randn(n, 2)*0.1  # add some noise
    return (X, y)


def gauss(N0=1000, N1=None, r1=1, stddev=0.01):
    N = N0
    if N1 is not None:
        N += N1
    X = (np.random.rand(N, 2)*2-1)*r1
    y = np.random.rand(1) < \
        scipy.stats.norm.pdf(X[:, 0]*r1, 0, stddev) * \
        scipy.stats.norm.pdf(X[:, 1]*r1, 0, stddev) / \
        (scipy.stats.norm.pdf(1, 0, stddev)**2)
    return (X, y)


def gauss2(n=1000, r1=8, stddev=2):
    X = (np.random.rand(n, 2)*2-1)*r1
    y = np.random.rand(n) < \
        scipy.stats.norm.pdf(X[:, 0], 0, stddev) * \
        scipy.stats.norm.pdf(X[:, 1], 0, stddev) / \
        (scipy.stats.norm.pdf(1, 0, stddev)**2)
    return (X, y)


def hyperbola(N0=1000, N1=None, stddev=0.125):
    N = N0
    if N1 is None:
        N0 /= 2
        N1 = N0
    else:
        N += N1
    X = np.r_[np.random.rand(N0/2, 2)*0.5 + (0, 0.5),
              np.random.rand(N0/2, 2)*0.5 + (0.5, 0),
              np.random.rand(N1/2, 2)*0.5,
              np.random.rand(N1/2, 2)*0.5 + 0.5]
    C = np.r_[np.repeat(N0/float(N0+N1), N0), np.repeat(N1/float(N0+N1), N1)]
    y = np.zeros(N, int)
    for i in xrange(N):
        v = X[i, :]
        v = v - 0.5
        v = np.prod(v)
        v = v*10
        epsilon = stddev*np.random.randn(1) * C[i]
        v = v + epsilon
        y[i] = 0 if v < 0 else 1
    return (X, y)


def circle2(N0=1000, N1=None, rsq=None, stddev=0):
    if rsq is None:  # pi.r^2 = area => r = sqrt(area/pi)
        area = 4*N1/float(N0+N1)  # 4 is the area of the square
        rsq = area / np.pi
        N = N0+N1
    else:
        N = N0
    X = np.random.rand(N, 2)*2 - 1
    y = X[:, 0]**2 + X[:, 1]**2 <= rsq
    X[y == 1] += np.random.randn(np.sum(y), 2)*stddev
    return (X, y)


def xor(N0=1000, N1=None, stddev=0):  # mto parecido com o hyperbola
    if N1 is None:
        N0 = N0/2
        N1 = N0
    else:
        assert N1 % 4 == 0
    X = np.concatenate((np.random.rand(N0/2, 2)-(0, 1),
                        np.random.rand(N0/2, 2)-(1, 0),
                        np.random.rand(N1/2, 2) - 1 +
                        np.random.randn(N1/2, 2)*stddev,
                        np.random.rand(N1/2, 2) +
                        np.random.randn(N1/2, 2)*stddev), axis=0)
    for i in xrange(len(X)):
        for j in xrange(2):
            X[i, j] = max(X[i, j], -1)
            X[i, j] = min(X[i, j], +1)
    y = np.r_[np.zeros(N0), np.ones(N1)]
    return (X, y)


def easy(N=1000):  # easily separable
    X = np.concatenate((np.random.rand(N/2, 2)*(2, 1)-1,
                        np.random.rand(N/2, 2)*(2, 1)-(1, 0)), axis=0)
    y = np.concatenate((np.zeros(N/2), np.ones(N/2)), axis=0)
    return (X, y)


def slope(N=1000, slope=3):
    X1_0 = np.random.rand(N/2, 1)
    X2_0 = np.random.rand(N/2, 1)*X1_0*slope

    X1_1 = np.random.rand(N/2, 1)
    X2_1 = (1-np.random.rand(N/2, 1)*(1-X1_1))*slope

    X = np.append(np.append(X1_0, X2_0, 1),
                  np.append(X1_1, X2_1, 1), 0)
    y = np.append(np.zeros(N/2), np.ones(N/2), 0)
    return (X, y)


def spirals(N=1000):
    # uma circunferencia mas cujo raio cresce em proporcao ao angulo
    # x(t) = t.sin(t), y(t) = t.cos(t)
    t = np.random.rand(N/2) * 2.1*2*np.pi
    X = np.hstack((t * np.vstack((np.sin(t), np.cos(t))),
                   t * np.vstack((np.sin(t+np.pi), np.cos(t+np.pi)))))
    X = np.transpose(X)
    y = np.append(np.zeros(N/2), np.ones(N/2), 0)
    return (X, y)


def halfkernel(N=1000):
    # meia circunferencia
    t = np.random.rand(N/2) * np.pi
    X = np.hstack((1 * np.vstack((np.sin(t), np.cos(t))),
                   2 * np.vstack((np.sin(t), np.cos(t)))))
    X = np.transpose(X)
    y = np.append(np.zeros(N/2), np.ones(N/2), 0)
    return (X, y)


def torned(N=1000):
    X1 = np.random.rand(N)
    X2 = np.random.rand(N) + (1-X1)
    y = np.asarray([1 if x1 > 0.4 and x1 < 0.6 else 0 for x1 in X1])
    X = np.transpose(np.vstack((X1, X2)))
    return (X, y)


def random(N=1000):
    X1 = np.random.rand(N)
    X2 = np.random.rand(N)
    y = np.random.randint(2, size=N)
    X = np.transpose(np.vstack((X1, X2)))
    return (X, y)


fns = ('circle', 'gauss', 'xor', 'hyperbola', 'easy', 'slope', 'spirals',
       'halfkernel', 'torned', 'random')

if __name__ == "__main__":
    import matplotlib.pyplot as plot
    plot.clf()
    for idx, s in enumerate(fns):
        plot.subplot(np.ceil(len(fns)/4.), 4, idx+1)
        X, y = locals()[s](500)
        plot.plot(
            X[y == 0, 0], X[y == 0, 1], 'bo',
            X[y == 1, 0], X[y == 1, 1], 'ro',
            markersize=4)

        plot.title(s)
        plot.xticks([])
        plot.yticks([])

    plot.axis('tight')
    plot.show()
