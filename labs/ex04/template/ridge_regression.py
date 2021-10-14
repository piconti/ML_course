# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import *


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    A = np.dot(tx.T, tx)
    A2 = A + np.identity(A.shape[0])*2*tx.shape[0]*lambda_
    w_star = np.linalg.solve(A2, np.dot(tx.T, y))
    return compute_mse(y, tx, w_star), w_star
