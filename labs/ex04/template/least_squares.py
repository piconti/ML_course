# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import *


def least_squares(y, tx):
    """calculate the least squares."""
    w_star = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    return compute_mse(y, tx, w_star), w_star