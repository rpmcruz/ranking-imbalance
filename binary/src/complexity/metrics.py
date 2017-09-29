# -*- coding: utf-8 -*-

import samples
import numpy as np
import itertools


def overlap1(X, y):  # rate of positives having negative as closest neighbor
    diff = 0
    for i in xrange(len(X)):
        # (cannot be optimized because we need correct indices)
        if y[i] == 1:
            other = 0
            dmin = np.inf
            for j in xrange(len(X)):
                if i != j:
                    d = np.linalg.norm(X[i] - X[j])
                    if d < dmin:
                        dmin = d
                        other = y[j]
            diff += (other == 0)
    return float(diff) / np.sum(y == 1)


def overlap2(X, y):  # how many different until the same
    diff = 0
    for i in xrange(len(X)):
        if y[i] == 1:
            dmin = np.inf
            for j in xrange(len(X)):
                if y[j] == 1 and i != j:
                    d = np.linalg.norm(X[i] - X[j])
                    if d < dmin:
                        dmin = d
            other = 0
            for j in xrange(len(X)):
                if y[j] == 0 and i != j:
                    d = np.linalg.norm(X[i] - X[j])
                    if d < dmin:
                        other += 1
            diff += other
    return float(diff) / np.sum(y == 1)


# É para normais...
def bhattacharya1(X, y):
    X1 = X[y == 0]
    X2 = X[y == 1]
    mu1 = np.mean(X1, 0)
    mu2 = np.mean(X2, 0)
    sigma1 = np.cov(X1, rowvar=0)
    sigma2 = np.cov(X2, rowvar=0)
    return (1./8.) * np.dot(np.dot((mu2-mu1).T, np.linalg.inv(sigma1 + sigma2)), mu2-mu1) + 0.5*np.log(np.linalg.det(sigma1+sigma2) / (np.sqrt(np.linalg.det(sigma1)*np.linalg.det(sigma2))))


def partition(X, B):
    assert B > 0
    D = X.shape[1]
    b = np.zeros((D, B))
    amin = np.amin(X, 0)
    amax = np.amax(X, 0)
    for j in xrange(D):
        b[j] = np.linspace(amin[j], amax[j], B, False)
    K = np.zeros(X.shape[0], int)
    for i, x in enumerate(X):
        for j in xrange(D):
            K[i] += (np.sum(x[j] >= b[j])-1) * (B**j)
    return K


def bhattacharya2(X, y):
    # bhattacharya coefficient for samples
    # https://en.wikipedia.org/wiki/Bhattacharyya_distance
    nk = 10
    K = partition(X, nk)
    n = len(X)
    ret = 0
    for k in xrange((nk+1)**X.shape[1]):
        yK = y[K == k]
        p0 = np.sum(yK == 0) / float(n)
        p1 = np.sum(yK == 1) / float(n)
        ret += np.sqrt(p0*p1)
    return ret


def fisher(X, y):
    maxf1 = 0
    for j in xrange(X.shape[1]):
        X0j = X[y == 0, j]
        X1j = X[y == 1, j]
        mu1 = np.mean(X0j)
        mu2 = np.mean(X1j)
        sigma1 = np.var(X0j)
        sigma2 = np.var(X1j)
        f1 = (mu1-mu2)**2 / (sigma1+sigma2)
        maxf1 = max(f1, maxf1)
    return maxf1


def overlap_region(X, y):  # F2 (Singh, 2003)
    ret = 1
    for j in xrange(X.shape[1]):
        X0 = X[y == 0, j]
        X1 = X[y == 1, j]
        min_overlap = max(np.amin(X0), np.amin(X1))
        max_overlap = min(np.amax(X0), np.amax(X1))
        overlap = max_overlap - min_overlap
        if overlap < 0:
            overlap = 0
        overlap = overlap / (np.amax(X) - np.amin(X))
        ret *= overlap
    return ret ** (1./X.shape[1])


def feature_efficiency(X, y):
    maxfrac = 0
    for j in xrange(X.shape[1]):
        X0 = X[y == 0, j]
        X1 = X[y == 1, j]
        min_overlap = max(np.amin(X0), np.amin(X1))
        max_overlap = min(np.amax(X0), np.amax(X1))
        if max_overlap > min_overlap:
            z = np.logical_or(X[:, j] < min_overlap, X[:, j] > max_overlap)
            n = np.sum(z)
            frac = float(n) / float(len(y))
            maxfrac = max(maxfrac, frac)
    return maxfrac


def minimum_span_tree(X, y):  # is this well implemented?
    G = np.zeros((len(X), len(X)))
    for i, x in enumerate(X):
        for j, z in enumerate(X):
            G[i, j] = np.linalg.norm(x-z)
    import scipy.sparse
    G = scipy.sparse.csr_matrix(G)
    G = scipy.sparse.csgraph.minimum_spanning_tree(G)
    G = G.toarray()
    n = 0
    for i, g in enumerate(G):
        if np.any(y[i] != y[g > 0]):
            n += 1
    return float(n) / float(len(y)-n)


def inter_intra_cluster(X, Y):
    # formulas based on Anastasiu et al (2013)
    X0 = X[y == 0]
    X1 = X[y == 1]
    mu = np.mean(X, 0)
    mu0 = np.mean(X0, 0)
    mu1 = np.mean(X1, 0)
    Sw0 = 0  # Sw inter-cluster
    Sw1 = 0
    for x0 in X0:
        Sw0 += np.dot((x0-mu0), (x0-mu0).T)
    for x1 in X1:
        Sw1 += np.dot((x1-mu1), (x1-mu1).T)
    Sb0 = np.dot((mu0-mu), (mu0-mu).T)  # Sb intra-cluster
    Sb1 = np.dot((mu1-mu), (mu1-mu).T)
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    Sw = (n0*Sw0 + n1*Sw1) / (n0+n1)
    Sb = (n0*Sb0 + n1*Sb1)
    return Sw/Sb


def inter_intra_cluster2(X, Y):
    import sklearn
    dists = sklearn.metrics.pairwise.pairwise_distances(X)
    A = [sklearn.metrics.cluster.unsupervised._nearest_cluster_distance(
        dists[i], Y, i) for i in xrange(len(X))]
    B = [sklearn.metrics.cluster.unsupervised._intra_cluster_distance(
        dists[i], Y, i) for i in xrange(len(X))]
    return np.mean(B)/np.mean(A)


def space_covered(X, Y):  # nao tenho certeza
    ps = 0
    for x1, y1 in itertools.izip(X, Y):
        mindist = np.inf
        for x2, y2 in itertools.izip(X, Y):
            if y1 != y2:
                mindist = min(mindist, np.linalg.norm(x1-x2))
        n = 0
        for x2, y2 in itertools.izip(X, Y):
            if y1 == y2:
                if np.linalg.norm(x1-x2) < mindist:
                    n += 1
        ps += n
    return float(ps)/(len(X)**2)


"""
# PRISM metrics (Singh, 2003)

def under_curve(xs, ys):
    xs = np.append(xs, (1, 0))
    ys = np.append(ys, (0, 0))

    ret = 0
    for i in xrange(len(xs)):
        x0 = xs[i]
        y0 = ys[i]
        if i+1 < len(xs):
            x1 = xs[i+1]
            y1 = ys[i+1]
        else:
            x1 = xs[0]
            y1 = ys[0]
        ret += x0*y1 - y0*x1
    return np.abs(0.5*ret)


def ASh(X, y):
    Shs = []
    for B in xrange(0, 32):
        K = partition(X, B+1)
        D = X.shape[1]
        Sh = 0
        for l in xrange((B+1)**D):  # forp0 each cell
            XK = float(np.sum(K == l))
            if XK > 0:
                XK0 = float(np.sum(np.logical_and(K == l, y == 0)))
                p0 = XK0 / XK
                assert p0 >= 0 and p0 <= 1, p0
                p1 = 1-p0
                Shl = np.sqrt(2*((p0-0.5)**2+(p1-0.5)**2))
                Sh += Shl * (XK / float(len(X)))
        w = 1./(2**B)
        Shs.append(Sh * w)
        assert Sh >= 0 and Sh <= 1.001, Sh
    return under_curve(np.linspace(0, 1, 32), Shs) / 0.0483870967517


def ASnn(X, y):  # implementação incompleta...
    Shs = []
    for B in xrange(0, 32):
        K = partition(X, B+1)
        D = X.shape[1]
        Sh = 0
        for l in xrange((B+1)**D):  # forp0 each cell
            XK = X[K == l]
            for ci in xrange(2):
                lambda_il = np.logical_and(K == l, y == ci)
                for k in xrange(min(11, lamda_il)+1):
                    knn = NearestNeighbors(n_neighbors=k).fit(XK)
                    for x, _y in itertools.izip(XK, y[K == l]):
                        if _y == c:
                            nni = knn.kneighbors(x, return_distance=False)[0]
                            pk = np.sum(_y[nni] == c) / float(k)
                            # TODO
        Shs.append(Sh * w)
        assert Sh >= 0 and Sh <= 1.001, Sh
    return under_curve(np.linspace(0, 1, 32), Shs) / 0.0483870967517
"""


def silhouette_positive(X, y):
    import sklearn.metrics
    dists = sklearn.metrics.pairwise.pairwise_distances(X, metric='euclidean',
                                                        n_jobs=4)
    s = sklearn.metrics.silhouette_samples(dists, y, 'precomputed')
    s = np.mean(s[y == 1])
    return 1-np.abs(s)


if __name__ == "__main__":
    import matplotlib.pyplot as plot
    for idx, fn in enumerate(samples.fns):
        X, y = getattr(samples, fn)(1000)
        plot.plot(
            X[y == 0, 0], X[y == 0, 1], 'bo',
            X[y == 1, 0], X[y == 1, 1], 'ro',
            markersize=4)
        plot.title(fn)
        plot.xticks([])
        plot.yticks([])
        plot.axis('tight')
        plot.show()

        print '== disjunct =='
        print 'silhouette: %.2f' % silhouette_positive(X, y)
        print 'inter-intra cluster: %.2f' % inter_intra_cluster(X, y)
        print 'inter-intra cluster2: %.2f' % inter_intra_cluster2(X, y)
        print '== overlap =='
        print 'overlap1: %.2f' % np.mean(overlap1(X, y))
        print 'overlap2: %.2f' % np.mean(overlap2(X, y))
        print 'bhattacharya1: %.2f' % bhattacharya1(X, y)
        print 'bhattacharya2: %.2f' % bhattacharya2(X, y)
        print 'Fisher-discriminant: %.2f' % fisher(X, y)
        print 'F2 overlap: %.2f' % overlap_region(X, y)
        print 'F3 efficiency: %.2f' % feature_efficiency(X, y)
        print 'minimum span: %.2f' % minimum_span_tree(X, y)
        print 'T1 space covered: %.2f' % space_covered(X, y)
