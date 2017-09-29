# -*- coding: utf-8 -*-

# Algoritmos de resampling:
# * undersample: escolhe aleatoriamente sem repetição alguns dos pontos
# * oversample: mantém os primeiros e escolhe aleatoriamente com repetição
#               o resto
# * smote: oversample usando (Chawla, 2002), usa vizinhos. A nossa versão é
#          um pouco diferente: a original funciona apenas para múltiplos de
#          100.
# * msmote: variação em que:
#      all neighbors = 1 -> safe -> qualquer vizinho
#      all neighbors = 0 -> noise -> nada
#      else              -> border -> nearest neighbor

from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


def smote(T, unusedX, unusedy, N, k):
    if N == 0:
        return np.empty((0, T.shape[1]))
    knn = NearestNeighbors(n_neighbors=min(len(T), k)).fit(T)
    S = np.zeros((N, T.shape[1]))
    for n in range(N):
        i = np.random.randint(len(T))
        nn = knn.kneighbors(T[[i]], return_distance=False)
        # repeat until knn returns neighbor different than T[i]
        while True:
            nni = np.random.choice(nn[0])
            if nni != i:
                break
        dif = T[nni] - T[i]
        gap = np.random.random()
        S[n] = T[i] + gap*dif
    return S


def msmote(T, X, y, N, k):
    if N == 0:
        return np.empty((0, T.shape[1]))

    # train with all data
    knn = NearestNeighbors(n_neighbors=min(len(X), k)).fit(X)

    # msmote rules
    # 0 noise, 1 border, 2 safe
    types = np.zeros(len(T))
    nn = knn.kneighbors(T, return_distance=False)
    for i in range(len(T)):
        n1 = np.sum(y[nn[i]])
        if n1 == k:  # safe
            types[i] = 2
        elif n1 == 0:  # noise
            types[i] = 0
        else:
            types[i] = 1
    if np.sum(types == 0) == len(T):
        # all points are noise - that can't be right
        types = np.ones(len(T))

    # re-train kNN using only the target class now
    knn = NearestNeighbors(n_neighbors=min(len(T), k)).fit(T)

    S = np.zeros((N, T.shape[1]))
    n = 0
    while n < N:
        i = np.random.randint(len(T))
        # msmote application
        if types[i] == 1:
            nn = knn.kneighbors(T[[i]], return_distance=False)
            # repeat until knn returns neighbor different than T[i]
            while True:
                nni = np.random.choice(nn[0])
                if nni != i:
                    break
        elif types[i] == 2:
            dist, nn = knn.kneighbors(T[[i]])
            nni = nn[0][np.argmin(dist)]
        else:
            continue
        nn = knn.kneighbors(T[[i]], return_distance=False)
        # repeat until knn returns neighbor different than T[i]
        while True:
            nni = np.random.choice(nn[0])
            if nni != i:
                break
        dif = T[nni] - T[i]
        gap = np.random.random()
        S[n] = T[i] + gap*dif
        n += 1
    return S


def balanced_smote(X, y, knn, smote_fn):
    _X = np.empty((0, X.shape[1]))
    _y = np.empty(0, dtype=int)
    # classes must be [0,k[
    K = np.max(y)+1

    count = np.bincount(y)
    max_count = int(np.ceil(np.median(count)))

    for k in range(K):
        if count[k] == 0:
            continue
        x = X[[y == k]]
        if count[k] > max_count:
            x = x[np.random.choice(np.arange(count[k]), max_count, False)]
            _X = np.r_[_X, x]
        elif count[k] < max_count:
            deficit = max_count - count[k]
            xnew = smote_fn(x, X, (y == k).astype(int), deficit, knn)
            _X = np.r_[_X, x, xnew]
        else:  # count[k] == max_count
            _X = np.r_[_X, x]
        _y = np.r_[_y, np.repeat(k, max_count)]
    return _X, _y


K_NEIGHBORS = 5

class SMOTE(BaseEstimator, ClassifierMixin):
    def __init__(self, model, msmote=False):
        super(SMOTE, self).__init__()
        self.model = model
        self.msmote = msmote

    def fit(self, X, y):
        self.classes_ = np.arange(np.amax(y)+1)
        fn = msmote if self.msmote else smote
        X, y = balanced_smote(X, y, K_NEIGHBORS, fn)
        return self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


class MSMOTE(SMOTE):
    def __init__(self, model):
        SMOTE.__init__(self, model, True)


if __name__ == '__main__':
    import matplotlib.pyplot as plot
    plot.ioff()

    def gauss(n=1000, r=8, stddev=2):
        import scipy.stats
        X = (np.random.rand(n, 2)*2-1)*r
        y = np.random.rand(n) < \
            scipy.stats.norm.pdf(X[:, 0], 0, stddev) * \
            scipy.stats.norm.pdf(X[:, 1], 0, stddev) / \
            (scipy.stats.norm.pdf(1, 0, stddev)**2)
        return (X, y)

    X, y = gauss(250)

    # normal
    plot.subplot(1, 4, 1)
    plot.plot(
        X[y == 0, 0], X[y == 0, 1], 'bo',
        X[y == 1, 0], X[y == 1, 1], 'ro',
        markersize=4)
    plot.title('Normal')

    factor = 4

    # smote
    Xn = smote(X, np.sum(y)*factor, 8)
    plot.subplot(1, 4, 2)
    plot.plot(
        X[y == 0, 0], X[y == 0, 1], 'bo',
        Xn[:, 0], Xn[:, 1], 'go',
        X[y == 1, 0], X[y == 1, 1], 'ro',
        markersize=4)
    plot.title('SMOTE %dx' % factor)

    # msmote
    Xn = msmote(X, np.sum(y)*factor, 8)
    plot.subplot(1, 4, 3)
    plot.plot(
        X[y == 0, 0], X[y == 0, 1], 'bo',
        Xn[:, 0], Xn[:, 1], 'go',
        X[y == 1, 0], X[y == 1, 1], 'ro',
        markersize=4)
    plot.title('MSMOTE %dx' % factor)

    # normal
    X, y = gauss(250*factor)
    plot.subplot(1, 4, 4)
    plot.plot(
        X[y == 0, 0], X[y == 0, 1], 'bo',
        X[y == 1, 0], X[y == 1, 1], 'ro',
        markersize=4)
    plot.title('Normal %dx' % factor)

    plot.show()
