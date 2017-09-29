# -*- coding: utf-8 -*-

import numpy as np


def load_csv(filename):
    with open(filename, 'r') as f:
        counts = [f.readline().count(x) for x in (' ', ',')]
        delimiter = max(zip((' ', ','), counts), key=lambda x: x[1])[0]
    data = np.loadtxt(filename, delimiter=delimiter)
    X = data[:, 0:-1]
    y = data[:, -1]
    return (X, y)
