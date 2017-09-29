from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np

BAG_STRATEGY = 'voting'  # voting, max-proba


def smart_choice_without_replacement(a, size):
    ret = np.repeat(a, int(size/len(a)))
    plus = np.random.choice(a, size % len(a), False)
    return np.concatenate((ret, plus))


# our balanced bagging implementation

class BalancedBag(BaseEstimator, ClassifierMixin):
    def __init__(self, model, bags, bag_size):
        self.model = model
        self.bags = bags
        self.bag_size = bag_size

    def fit(self, X, y):
        #self.classes_ = np.unique(y)  # required by sklearn
        self.classes_ = np.arange(np.max(y))  # required by sklearn

        #print('classes:', self.classes_)
        count = np.bincount(y)
        self.models = [None] * self.bags
        if isinstance(self.bag_size, str):
            bag_size = int(np.ceil(getattr(np, self.bag_size)(count)))
        else:
            bag_size = self.bag_size
        for i in range(self.bags):
            _X = np.empty((0, X.shape[1]))
            _y = np.empty(0, int)
            for k in self.classes_:
                if count[k] == 0:
                    continue
                ix = smart_choice_without_replacement(
                    np.where(y == k)[0], bag_size)
                Xk = X[ix]
                _X = np.r_[_X, Xk]
                _y = np.r_[_y, np.repeat(k, bag_size)]
            m = clone(self.model)
            self.models[i] = m.fit(_X, _y)
        return self

    def predict(self, X):
        if BAG_STRATEGY == 'voting':
            votes = np.zeros((len(X), len(self.classes_)), int)
            for model in self.models:
                yp = model.predict(X)
                for i, k in enumerate(yp):
                    votes[i, k] += 1
            return np.argmax(votes, 1)

        res = np.zeros((len(X), len(self.classes_)))
        for model in self.models:
            try:
                pp = model.predict_proba(X)
            except:
                pp = model.decision_function(X)
            res += pp
        return np.argmax(res, 1)


if __name__ == '__main__':
    def make_imbalance(X, y, ratio):
        _X = np.empty((0, X.shape[1]))
        _y = np.empty(0, int)
        ks = np.unique(y)
        ratios = np.linspace(ratio, 1, len(ks))
        for k in ks:
            n = int(np.sum(y == k) * ratios[k])
            idx = np.random.choice(np.where(y == k)[0], n, False)
            _X = np.r_[_X, X[idx]]
            _y = np.r_[_y, y[idx]]
        return _X, _y

    from sklearn import datasets
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    import sys

    #X, y = make_imbalance(*datasets.load_digits(10, True), 0.1)
    X, y = make_imbalance(*datasets.make_classification(
        10000, n_informative=3, n_classes=4), 0.1)
    count = np.bincount(y)
    print('count:', count)
    models = [
        ('imbalance', LinearSVC()),
        ('weights', LinearSVC(class_weight='balanced')),
        ('balanced bag', BalancedBag(LinearSVC(), 10, np.min(count))),
    ]
    scores_fn = [
        ('acc', accuracy_score),
        ('f1', lambda y, yp: f1_score(y, yp, average='weighted')),
        ('f1-0', lambda y, yp: f1_score(y == 0, yp == 0)),
    ]
    scores_sum = np.zeros((len(models), len(scores_fn)))
    scores_sum2 = np.zeros((len(models), len(scores_fn)))
    folds = 10
    for fold, (tr, ts) in enumerate(StratifiedKFold(folds).split(X, y)):
        for i, (_, model) in enumerate(models):
            progress = (i+len(models)*fold)/(folds*len(models))
            sys.stdout.write('\r%2d%%' % (100*progress))
            model.fit(X[tr], y[tr])
            yp = model.predict(X[ts])
            s = np.asarray([s(y[ts], yp) for _, s in scores_fn])
            scores_sum[i] += s
            scores_sum2[i] += s**2
    sys.stdout.write('\r             \r')
    scores_mean = scores_sum / folds
    scores_sdev = np.sqrt(scores_sum2 - (scores_sum**2)/folds)
    print(' '*16 + ' '.join(['%-10s' % s for s, _ in scores_fn]))
    for i, (name, _) in enumerate(models):
        print(
            '%15s ' % name +
            ' '.join([
                '%.2f (%.1f)' % r for r in zip(
                    scores_mean[i], scores_sdev[i])]))
