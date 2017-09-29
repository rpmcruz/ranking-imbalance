from sklearn.base import BaseEstimator, ClassifierMixin
from osvm import OrdinalSVM
from ranksvm import RankSVM
from threshold_kelwin import decide_thresholds
from threshold import threshold_to_class
import numpy as np

class RankOSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold_strategy):
        self.model = model
        self.threshold_strategy = threshold_strategy

    def fit(self, X, y):
        m = OrdinalSVM()
        m.K = np.max(y)+1
        Xr, yr = m.replicate_data(X, y+1)

        m = RankSVM(self.model).fit(Xr, yr)
        self.coef = self.model.coef_[0][:X.shape[1]]

        self.classes_ = np.arange(np.amax(y)+1, dtype=int)
        s = self.predict_proba(X)
        self.ths = decide_thresholds(
            s, y, len(self.classes_), self.threshold_strategy)
        return self

    def predict_proba(self, X):
        return -np.sum(self.coef*X, 1)

    def decision_function(self, X):
        return self.predict_proba(X)

    def predict(self, X):
        s = self.predict_proba(X)
        return threshold_to_class(s, self.ths)

