# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    #set seed
    np.random.seed(seed)
    
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    ratio_idx = int(ratio*len(y))
    train_y = shuffled_y[:ratio_idx]
    train_x = shuffled_x[:ratio_idx]
    test_y = shuffled_y[ratio_idx+1:]
    test_x = shuffled_x[ratio_idx+1:]
    return train_x, train_y, test_x, test_y
