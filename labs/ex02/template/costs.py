# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w, mae = False):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    defzlut loss: MSE, can return MAE
    """
    e = y - tx.dot(w)
    if(mae):
        return np.mean(np.abs(e))
    return 1/2*np.mean(e**2)