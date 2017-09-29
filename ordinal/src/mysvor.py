'''
Wrapper around svor[im/ex]
'''

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import os
import shutil
from datetime import datetime

CLEANUP = True
SILENT = True


class SVOR(BaseEstimator, ClassifierMixin):
    def __init__(self, cmd, C, kernel):
        self.cmd = cmd
        self.C = C
        self.kernel = kernel

    def fit(self, Xtr, ytr):
        #self.classes_ = np.unique(ytr)  # required by sklearn
        # for our datasets, this makes more sense:
        self.classes_ = np.arange(np.amax(ytr)+1, dtype=int)

        self.Xtr = Xtr
        self.ytr = ytr + 1  # svor wants [1,k]
        return self

    def predict(self, Xts):
        # train and predict

        dt = np.modf(datetime.today().timestamp())
        r = int(np.random.random()*100000)
        dirname = 'temp/%d_%s_kernel_%s_C_%f_%d_%d_%d' % (
            os.getpid(), self.cmd, self.kernel, self.C, int(dt[1]),
            int(dt[0]*1e12), r)
        os.mkdir(dirname)
        tr_filename = dirname + '/model_train.0'
        ts_filename = dirname + '/model_test.0'
        yp_filename = dirname + '/model_cguess.0'

        Xtr = np.c_[self.Xtr, self.ytr]
        fmt = ['%f']*self.Xtr.shape[1] + ['%d']
        np.savetxt(tr_filename, Xtr, fmt)
        np.savetxt(ts_filename, Xts, '%f')

        args = ['-Z 0', '-Co %f' % self.C]
        if self.kernel == 'rbf':
            pass
        else:
            args.append('-P 1')
        cmd = '%s/%s %s %s' % (self.cmd, self.cmd, ' '.join(args), tr_filename)
        if SILENT:
            cmd += ' >/dev/null 2>&1'
        else:
            print('cmd:', cmd)
        os.system(cmd)

        yp = np.loadtxt(yp_filename)
        yp -= 1 # convert back to [0,k[
        assert len(yp) == len(Xts)

        if CLEANUP:
            shutil.rmtree(dirname)
        return yp


class SVORIM(SVOR):
    def __init__(self, C, kernel):
        super().__init__('svorim', C, kernel)


class SVOREX(SVOR):
    def __init__(self, C, kernel):
        super().__init__('svorex', C, kernel)
