import numpy as np
import datetime
from implementations import *
from cross_validation import *


def generate_gamma(deg):
    """
    Creates an array containing gamma values   
    Input:
    - nb (int) : number of gammas to generate
    Output:    
    - d1 (ndarray) : array containing gamma values with values 10 times smaller than the previous value
    """
    d1 = [10**(-(i+1)) for i in range(deg)]
    return np.array(d1)

def generate_lambda(deg):
    """
    Creates an array containing lambdas values   
    Input:
    - nb (int) : number of lambdas to generate
    Output:    
    - d1 (ndarray) : array containing lambdas values with values 10 times smaller than the previous value
    """
    d1 = [10**(-(i+1)) for i in range(deg)]
    return np.array(d1)

def generate_degree(degree):
    """
    Creates an array containing degrees of the polynomial expansion   
    Input:
    - degree (int) : maximum degree of the polynomial expansion
    Output:    
    - deg (ndarray) : array containing degree values from j=1 to j=degree
    """
    deg = [i+1 for i in range(degree)]
    return np.array(deg)


"""Grid search with 1 hyperparameter"""

def get_best_parameters_1_param(degree, acc):
    """
    Gets the best hyperparameters for a model  
    Inputs:
    - degree (ndarray) : array containing the degree tested
    - acc (ndarray) : array containing the accuracies for each degree tested
    Outputs:    
    - acc[min_row] : best accuracy achieved during training
    - degree[min_row] : degree for which best accuracy was achieved
    """
    acc[np.isinf(acc)] = np.nan
    min_row = np.argmax(acc)
    return acc[min_row], degree[min_row]

def grid_search_1_param(y, tx, degree, type_="ls", max_iters=100, k_fold=10):
    """
    Performs grid search when there is one parameter that is the degree of the polynomial expansion   
    Inputs:
    - y (ndarray) : labels of the input data
    - tx (ndarray) : features of the input data
    - degree (ndarray) : array containing the degrees of the expansion features
    - type (string) : by default ls for least squares
    - max_iters (int) : by default to 100, maximum number of iterations allowed
    - k_fold (int) : by default to 10, number of k folds for cross validation
    Outputs:    
    - acc_mean (ndarray) : means of the accuracy obtained during cross validation for each degree tested 
    - acc_std (ndarray) : standard deviations of the accuracy obtained during cross validation for each degree tested
    """
    acc_mean = np.zeros(len(degree))
    acc_std = np.zeros(len(degree))
    for deg_n in degree:
            acc_mean[deg_n-1], acc_std[deg_n-1] = cross_val_mean(y, tx, k_fold, deg_n, type_)
    return acc_mean, acc_std



"""Grid search with 2 hyperparameters"""

def get_best_parameters_2_param(gamma, degree, acc):
    """
    Gets the best parameters when grid search on two parameters is performed  
    Input:
    - gamma (ndarray) : array containing the gammas to perform grid search on
    - degree (ndarray) : array containing the degrees of polynomial expansion to perform grid search on
    - acc (ndarray) : array containing the accuracies for each combination of gamma and degree
    Output:    
    - acc[min_row, min_col] : best accuracy achieved
    - gamma[min_row] : gamma for which best accuracy was achieved
    - degree[min_col] : degree of polynomial expansion for which best accuracy was achieved
    """
    acc[np.isinf(acc)] = np.nan
    min_row, min_col = np.unravel_index(np.argmax(acc), acc.shape)
    return acc[min_row, min_col], gamma[min_row], degree[min_col]


def grid_search_2_param(y, tx, gamma, degree, type_="gd", max_iters=100, k_fold=10):
    """
    Performs grid_search when two parameters are used
    Input:
    - y (ndarray) : labels of the input data
    - tx (ndarray) : features of the input data
    - gamma (ndarray) : array containing the gammas to perform grid search on
    - degree (ndarray) : array containing the degrees of polynomial expansion to perform grid search on
    - type (string) : type of the model to apply by default to gd for gradient descent
    - max_iters (int) : maximum number of iterations allowed by default to 100
    - k_fold (int) : number of k-folds for cross-validation by default to 10
    Output:    
    - acc_mean (ndarray) : array containing the means of the accurracies obtained during cross-validation for each combination of
                           hyperparameters
    - acc_std (ndarray) : array containing the standard deviations of the accuracies obtained during cross-validation for each
                          combination of hyperparameters
    """
    acc_mean = np.zeros((len(gamma), len(degree)))
    acc_std = np.zeros((len(gamma), len(degree)))
    for i, gamma_n in enumerate(gamma):
        for j, deg_n in enumerate(degree):
            if type_=="gd":
                acc_mean[i,j], acc_std[i,j] = cross_val_mean(y, tx, k_fold, deg_n, type_, gamma = gamma_n)
            elif type_=="sgd":
                acc_mean[i,j], acc_std[i,j] = cross_val_mean(y, tx, k_fold, deg_n, type_, gamma = gamma_n)
            elif type_=="rr":
                acc_mean[i,j], acc_std[i,j] = cross_val_mean(y, tx, k_fold, deg_n, type_, lambda_ = gamma_n)
            elif type_=="lr":
                acc_mean[i,j], acc_std[i,j] = cross_val_mean(y, tx, k_fold, deg_n, type_, gamma = gamma_n)
    return acc_mean, acc_std


"""Grid search with 3 hyperparameters"""

def get_best_parameters_3_param(gamma,  degree, acc):
    """
    Gets the best parameters when grid search on three parameters is performed  
    Input:
    - gamma (ndarray) : array containing the gammas to perform grid search on
    - lambda_ (ndarray) : array containing the lambdas to perform grid search on
    - degree (ndarray) : array containing the degrees of polynomial expansion to perform grid search on
    - acc (ndarray) : array containing the accuracies for each combination of gamma, degree and lambda_
    Output:    
    - acc[min_first, min_second, min_third] : best accuracy achieved
    - lambda_[min_first] : lambda for which best accuracy was achieved
    - gamma[min_second] : gamma for which best accuracy was achieved
    - degree[min_third] : degree of polynomial expansion for which best accuracy was achieved
    """
    acc[np.isinf(acc)] = np.nan
    min_row, min_col = np.unravel_index(np.argmax(acc), acc.shape)
    return acc[min_row, min_col], gamma[min_row], degree[min_col]


def grid_search_3_param(y, tx, lambda_, gamma, degree, type_="rlr", max_iters=100, k_fold=10):
    """
    Performs grid_search when three parameters are used
    Input:
    - y (ndarray) : labels of the input data
    - tx (ndarray) : features of the input data
    - lambda_ (ndarray) : array containing the lambdas to perform grid search on
    - gamma (ndarray) : array containing the gammas to perform grid search on
    - degree (ndarray) : array containing the degrees of polynomial expansion to perform grid search on
    - type (string) : type of the model to apply by default to rlr for gradient descent
    - max_iters (int) : maximum number of iterations allowed by default to 100
    - k_fold (int) : number of k-folds for cross-validation by default to 10
    Output:    
    - acc_mean (ndarray) : array containing the means of the accurracies obtained during cross-validation for each combination of
                           hyperparameters
    - acc_std (ndarray) : array containing the standard deviations of the accuracies obtained during cross-validation for each
                          combination of hyperparameters
    """
    acc_mean = np.zeros((len(lambda_), len(gamma), len(degree)))
    acc_std = np.zeros((len(lambda_), len(gamma), len(degree)))
    for i, lambda_n in enumerate(lambda_):
        for j, gamma_n in enumerate(gamma):
            for k, deg_n in enumerate(degree):
                if type_=="rlr":
                    acc_mean[i,j,k], acc_std[i,j,k] = cross_val_mean(y, tx, k_fold, deg_n, type_, gamma_n, lambda_n)
    return acc_mean, acc_std