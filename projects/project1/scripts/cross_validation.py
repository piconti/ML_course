import numpy as np
import matplotlib.pyplot as plt
from implementations import * 
from eda_preprocessing import *


def bool_labels(y):
    """
    Turn the y labels from values in {-1, 1} to values in {0, 1}.
    Inputs:
    - y (ndarray): binary labels with values {-1, 1}
    Outputs: 
    - by (ndarray): binary labels with values {0, 1}
    """
    by = [0 if i==-1 else 1 for i in y]
    return np.array(by)


def build_k_indices(y, k_fold, seed):
    """
    Build the indices for k-fold cross-validation.
    Inputs:
    - y (ndarray): binary labels the size of the input data to split
    - k_fold (int): number of subsets to divide the data in
    - seed (int): random seed to fix a specific random splitting
    Outputs: 
    - k_indices (ndarray): indices for splitting with shape [k_fold,len(y)/k_fold]
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_val_mean(y, x, k_fold, degree, type_, gamma = None, lambda_ = None, max_iter = 100):
    """
    Trains model k_fold times each with given hyperparameters and returns the mean and std of the accuracies.
    Inputs:
    - y (ndarray): input training binary labels
    - x (ndarray): input training data
    - k_fold (int): number of times to train the model
    - degree (int): degree of polynomial feature expansion to perform on x
    - type_ (str): model to train for this cross-validation
    - gamma (float): gamma parameter to train the model with if applicable
    - lambda_ (float): lambda parameter to train the model with if applicable
    - max_iter (int): number of iterations to perform during training, default to 100 during cross-validation
    Outputs: 
    - mean_accs_test (float): mean of the test accuracies obtained over k_fold runs
    - std_accs_test (float): standard deviation of the test accuracies obtained over k_fold runs
    """
    k_indices = build_k_indices(y, k_fold, seed=12)
    accs_test = np.zeros(k_fold)
    for k in range(k_fold):
        accs_test[k] = cross_validation(y, x, k_indices, k, degree, type_, gamma, lambda_, max_iter)
    return np.mean(accs_test), np.std(accs_test)


def cross_validation(y, x, k_indices, k, degree, type_ , gamma = None, lambda_ = None, max_iter = 100):
    """
    Train kth subset of x with specified model and hyperparameters, return the test accuracy obtained.
    Inputs:
    - y (ndarray): input training binary labels
    - x (ndarray): input training data
    - k_indices (ndarray): random indices corresponding to each of the k_fold subsets to train on
    - k (int): index of set of indices to keep for this specific run
    - degree (int): degree of polynomial feature expansion to perform on x
    - type_ (str): model to train for this run
    - gamma (float): gamma parameter to train the model with if applicable
    - lambda_ (float): lambda parameter to train the model with if applicable
    - max_iter (int): number of iterations to perform during training, default to 100 during cross-validation
    Outputs: 
    - accs_te (float): test accuracies obtained for this subset and hyperparameters
    """
    test_x, test_y = x[k_indices[k]], y[k_indices[k]]
    train_indices = np.delete(k_indices,(k), axis=0)
    train_x, train_y = x[train_indices].reshape(-1, x.shape[-1]), y[train_indices].flatten()

    train_tx = build_poly(train_x, degree)
    test_tx = build_poly(test_x, degree)
    bool_y = bool_labels(train_y)
    
    initial_w = np.zeros(train_tx.shape[1])
    acc_te = 0
    
    if(type_ == "ls"):
        # least squares with normal equations
        w_star, loss_tr = least_squares(train_y, train_tx)
        acc_te = accuracy(test_y, test_tx, w_star)
    elif(type_ == "gd"):
        # linear regression with GD
        w_star, loss_tr = least_squares_GD(train_y, train_tx, initial_w, max_iter, gamma)
        acc_te = accuracy(test_y, test_tx, w_star)
    elif(type_ == 'sgd'):
        # linear regression with SGD
        w_star, loss_tr = least_squares_SGD(train_y, train_tx, initial_w, max_iter, gamma=gamma)
        acc_te = accuracy(test_y, test_tx, w_star)
    elif(type_ == 'rr'):
        # ridge regression with normal equations
        w_star, loss_tr = ridge_regression(train_y, train_tx, lambda_)
        acc_te = accuracy(test_y, test_tx, w_star)
    elif(type_ == 'lr'):
        # logistic regression with GD
        initial_w = np.zeros((train_tx.shape[1], 1))
        w_star, loss_tr = logistic_regression(bool_y, train_tx, initial_w, max_iter, gamma)
        acc_te = accuracy(test_y, test_tx, w_star)
    elif(type_ == 'rlr'):
        # regularized logistic regression with GD
        initial_w = np.zeros((train_tx.shape[1], 1))
        w_star, loss_tr = reg_logistic_regression(bool_y, train_tx, lambda_, initial_w, max_iter, gamma)
        acc_te = accuracy(test_y, test_tx, w_star)
    
    return acc_te