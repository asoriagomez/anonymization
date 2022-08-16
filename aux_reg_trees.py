

import numpy as np
import matplotlib.pyplot as plt

# Just a definition to plot deviance tests

def deviance_plot(est, X_test, y_test, ax=None, label='', train_color='#2c7bb6', test_color='#d7191c', alpha=1.0):
    """Deviance plot for ``est``, use ``X_test`` and ``y_test`` for test error. """
    n_estimators=len(est.estimators_)
    test_dev = np.empty(n_estimators)
    for i, pred in enumerate(est.staged_predict(X_test)):
        test_dev[i] = est.loss_(y_test, pred)

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

    ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label, linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color, label='Train %s' % label, linewidth=2, alpha=alpha)

    ax.set_ylabel('Error')
    ax.set_xlabel('n_estimators')
    return test_dev, ax



def fmt_params(params):
    return ", ".join("{0}={1}".format(key, val) for key, val in params.items())