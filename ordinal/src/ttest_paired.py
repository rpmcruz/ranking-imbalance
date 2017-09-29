import numpy as np
import scipy.stats


def ttest_paired(x1, x2):  # H0: x1 = x2, H1: x1 < x2
    if len(x1) != len(x2):
        print('ttest_paired warning: comparing different sizes:',
              len(x1), len(x2))
        return 1
    x = np.asarray(x1) - np.asarray(x2)
    mu = np.mean(x)
    stderr = np.sqrt(np.var(x)/len(x))
    if stderr == 0:
        return 1
    t = mu / stderr
    df = len(x)-1
    return scipy.stats.t.cdf(t, df)
