'''
Wrapper around svmlight
'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import dump_svmlight_file
import numpy as np
import os
from datetime import datetime
import shutil

FAST_RANK = True
SILENT = True


class SVMLight(BaseEstimator, ClassifierMixin):
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.dirname = None

    def __del__(self):
        if self.dirname:
            shutil.rmtree(self.dirname)

    def get_dirname(self):
        cmd = 'svmrank' if FAST_RANK else 'svmlight'
        dt = np.modf(datetime.today().timestamp())
        return 'temp/%d_%s_kernel_%s_C_%f_%d_%d' % (
            os.getpid(), cmd, self.kernel, self.C, int(dt[1]),
            int(dt[0]*1e12))

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # required by sklearn

        self.dirname = self.get_dirname()
        os.mkdir(self.dirname)
        tr_filename = self.dirname + '/train'
        model_filename = self.dirname + '/model'

        qid = np.ones(len(X))
        dump_svmlight_file(X, y, tr_filename, False, query_id=qid)

        args = [
            '-t %d' % (2 if self.kernel == 'rbf' else 0),
            '-c %.3f' % self.C,
            '-# 1000',  # default: 1e5
            #'-e 0.1', # default: 0.001 (TEMP)
        ]
        args = ' '.join(args)
        if FAST_RANK:
            cmd = 'svm_rank/svm_rank_learn %s %s %s' % (
                args, tr_filename, model_filename)
        else:
            cmd = 'svm_light/svm_learn -z p %s %s %s' % (
                args, tr_filename, model_filename)
        if SILENT:
            cmd += ' >/dev/null 2>&1'
        os.system(cmd)
        return self

    def predict_proba(self, X):
        y = qid = np.ones(len(X))

        model_filename = self.dirname + '/model'
        ts_filename = self.dirname + '/test'
        yp_filename = self.dirname + '/pred'

        dump_svmlight_file(X, y, ts_filename, False, query_id=qid)
        if FAST_RANK:
            cmd = 'svm_rank/svm_rank_classify %s %s %s' % (
                ts_filename, model_filename, yp_filename)
        else:
            cmd = 'svm_light/svm_classify %s %s %s' % (
                ts_filename, model_filename, yp_filename)
        if SILENT:
            cmd += ' >/dev/null 2>&1'
        os.system(cmd)
        yp = np.loadtxt(yp_filename)
        return yp
