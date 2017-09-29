# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plot


def plot_decision_surface(model, X, y, plot_step=0.02):
    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.asarray(Z)
    Z = Z.reshape(xx.shape)

    plot.contourf(xx, yy, Z, colors=('white', 'white', 'lightgray', 'white'))
    plot.plot(X[y == 0, 0], X[y == 0, 1], 'k.', markersize=0.5)
    plot.plot(X[y == 1, 0], X[y == 1, 1], 'k.', markersize=2.0)


def latex_style(hratio, wratio=1):
    fig_width_pt = wratio*236.84843  # get from latex: \the\textwidth
    inches_per_pt = 1.0/72.27  # inches to pt
    w = (fig_width_pt*inches_per_pt)*1
    figsize = (w, w*hratio)
    rc_latex = {
        # lines and points
        'lines.linewidth': 0.75,
        'lines.markersize': 3,
        # font
        'font.family': 'serif',
        # LaTeX default is 10pt font...
        'text.fontsize': 8,
        'axes.labelsize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'savefig.dpi': 250,
        'figure.figsize': figsize,
    }
    import matplotlib
    matplotlib.rcParams.update(rc_latex)
