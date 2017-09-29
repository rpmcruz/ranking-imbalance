from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
import collections
import numpy as np


class OrdinalSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None, C=1, h=1, s=1):
        super(OrdinalSVM, self).__init__()
        self.model = model
        self.C = C
        self.h = h
        self.s = s

    def fit(self, X, y):
        y += 1  # this code assumes [1,K]

        self.K = np.max(y)
        Xr, yr = self.replicate_data(X, y)
        if not self.model:
            self.model = LinearSVC(C=self.C, fit_intercept=False)  #, random_state=42)
        self.model.fit(Xr, yr)

        self.coef = np.array(self.model.coef_[0][: len(X[0])])
        self.b = -self.model.coef_[0][len(X[0]):] * self.h
        return self

    def replicate_data(self, Xaux, y):
        def u(v, K):
            ret = np.zeros((1, K - 1))
            ret[0][v - 1] = self.h
            return ret

        self.means = np.mean(Xaux, axis=0)
        self.stds = np.std(Xaux, axis=0)
        X = (Xaux - self.means) / (self.stds+1e-12)

        s = self.s
        K = self.K

        # First class (and first section)
        X1 = X[y == 1]
        n1 = X1.shape[0]

        Xleft = X[np.logical_and(2 <= y, y <= min(K, 1 + s))]
        nleft = Xleft.shape[0]

        X1 = np.vstack((X1, Xleft))
        X1 = np.hstack((X1, np.repeat(u(1, K), n1 + nleft, axis=0)))

        # Last class (and last section)
        XK = X[y == K]
        nK = XK.shape[0]

        Xright = X[np.logical_and(max(1, K - 1 - s + 1) <= y, y <= K - 1)]
        nright = Xright.shape[0]

        XK = np.vstack((Xright, XK))
        XK = np.hstack((XK, np.repeat(u(K - 1, K), nK + nright, axis=0)))

        Xret = np.vstack((X1, XK))
        yret = [(0, n1), (1, nleft), (0, nright), (1, nK)]

        # Inner classes
        for q in range(2, K):
            Xil = X[np.logical_and(max(1, q - s + 1) <= y, y <= q)]
            nil = Xil.shape[0]

            Xir = X[np.logical_and(q + 1 <= y, y <= min(K, q + s))]
            nir = Xir.shape[0]

            Xinner = np.vstack((Xil, Xir))
            Xinner = np.hstack((Xinner,
                                np.repeat(u(q, K), nil + nir, axis=0)))

            Xret = np.vstack((Xret, Xinner))
            yret += [(0, nil), (1, nir)]

        yret = np.hstack(np.zeros(n) if z == 0 else np.ones(n)
                         for z, n in yret)
        yret = yret.astype(np.int)
        return Xret, yret

    def predict(self, X):
        Xaux = (X - self.means) / (self.stds+1e-12)
        scores = np.dot(Xaux, self.coef)
        predictions = np.ones(scores.shape, dtype=np.int)

        for i, bi in enumerate(self.b):
            predictions[scores >= bi] = i + 2
        return predictions-1  # go back to [0,K[


class RankSVM(object):
    def __init__(self, C=1, s=-1):
        self.C = C
        self.s = s

    def fit(self, X, pairs):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)

        Xaux = (X - self.means) / self.stds
        Xdiff = Xaux[pairs[:, 0]] - Xaux[pairs[:, 1]]
        Xdiff = np.vstack((Xdiff, -Xdiff))
        ydiff = np.hstack((-np.ones(pairs.shape[0], dtype=np.int),
                           +np.ones(pairs.shape[0], dtype=np.int)))

        self.svm = LinearSVC(C=self.C, fit_intercept=False)  #, random_state=42)
        self.svm.fit(Xdiff, ydiff)
        self.coef = self.svm.coef_[0]
        return self

    def predict(self, X, pairs):
        Xaux = (X - self.means) / self.stds
        Xdiff = Xaux[pairs[:, 0]] - Xaux[pairs[:, 1]]
        return 2 * (np.dot(Xdiff, self.coef) <= 0) - 1

    def pairs_from_ordinal(self, y):
        def in_range(cl1, cl2):
            return (self.s < 0 and cl1 < cl2) or \
                (self.s >= 0 and cl1 < cl2 and cl2 <= self.s + cl1)

        def num_pairs():
            freq = collections.Counter(y)
            ret = np.sum([f1 * f2 for c1, f1 in freq.items()
                          for c2, f2 in freq.items() if in_range(c1, c2)])
            return ret

        ret = np.zeros((num_pairs(), 2), dtype=np.int)
        rangen = np.arange(y.shape[0])

        shift = 0
        classes = np.sort(np.unique(y))

        for cl1 in classes:
            x1 = rangen[y == cl1]
            for cl2 in classes:
                if in_range(cl1, cl2):
                    x2 = rangen[y == cl2]
                    next_shift = shift + x1.shape[0] * x2.shape[0]
                    ret[shift: next_shift, 0] = np.repeat(x1, x2.shape[0])
                    ret[shift: next_shift, 1] = np.tile(x2, x1.shape[0])
                    shift = next_shift

        return ret


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import csv
    import os

    def read_dataset(filename):
        f = open(filename)
        X = [map(float, x) for x in csv.reader(f, delimiter=',')]
        f.close()
        y = np.array([x[-1] for x in X]).astype(int)
        X = np.array([x[: -1] for x in X])
        return X, y

    X, y = read_dataset(os.sys.argv[1])
    y += 1

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    s = 1

    rsvm = RankSVM(C=100, s=s)
    p_train = rsvm.pairs_from_ordinal(y_train)
    p_test = rsvm.pairs_from_ordinal(y_test)

    rsvm.fit(X_train, p_train)
    print(' Pairwise', list(rsvm.coef / np.linalg.norm(rsvm.coef)))
    print('Acc', accuracy_score([-1] * len(p_test), rsvm.predict(X_test,
                                                                 p_test)))
    print()

    osvm = OrdinalSVM(C=100, h=1, s=s)
    osvm.fit(X_train, y_train)
    print('Ordinal-1', list(osvm.coef / np.linalg.norm(osvm.coef)))
    print('Acc', accuracy_score(y_test, osvm.predict(X_test)))
    print()

    osvm = OrdinalSVM(C=100, h=0.1, s=s)
    osvm.fit(X_train, y_train)
    print('Ordinal-h', list(osvm.coef / np.linalg.norm(osvm.coef)))
    print('Acc', accuracy_score(y_test, osvm.predict(X_test)))
    print()
    
    rsvm = RankSVM(C=100, s=s)
    rsvm.means = osvm.means
    rsvm.stds = osvm.stds
    rsvm.coef = osvm.coef
    print('OPairwise', list(rsvm.coef / np.linalg.norm(rsvm.coef)))
    print('Acc', accuracy_score([-1] * len(p_test), rsvm.predict(X_test,
                                                                 p_test)))
    print()
