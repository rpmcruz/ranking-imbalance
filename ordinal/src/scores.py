import numpy as np
from sklearn.metrics import mean_absolute_error


def maximum_mean_absolute_error(y, yp):
    y = np.asarray(y)
    yp = np.asarray(yp)
    klasses = np.unique(y)
    error = np.zeros(len(klasses))
    for i, k in enumerate(klasses):
        _y = y[y == k]
        _yp = yp[y == k]
        error[i] = mean_absolute_error(_y, _yp)
    return np.amax(error)
