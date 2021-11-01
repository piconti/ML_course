import numpy as np
from implementations import *


def summarize_features(data, include_nan = True, print_vals = True):
    """
    Prints statistical measures on each features and returns them    
    Input:
    - data (ndarray) : input set that contains the features we want to obtain measures on
    - include_nan (boolean) : by default to True, if True then nans should be included in the measurements else they are discarded
    - print_vals : by default to True, if True prints the values to the console else solely returns them
    Which obtained with split_data_jetnum()   
    Output:    
    - means (ndarray) : contains all the means of the features
    - stds (ndarray) : contains all the standard deviations of the features
    - medians (ndarray) : contains all the medians of the features
    - mins (ndarray) : contains all the minimums of the features
    - maxs (ndarray) : contains all the maximums of the features
    """
    if(include_nan):
        # calculate the different measures including nan values
        means = np.mean(data, axis=0)
        medians = np.median(data, axis=0)
        stds = np.std(data, axis=0)
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
    else:
        # calculate the different measures discarding nan values
        means = np.nanmean(data, axis=0)
        medians = np.nanmedian(data, axis=0)
        stds = np.nanstd(data, axis=0)
        mins = np.nanmin(data, axis=0)
        maxs = np.nanmax(data, axis=0)
    
    if(print_vals):
        # print the values obtained
        print()
        if(include_nan):
            print("summary variables, where nan values are not ignored:")
        else:
            print("summary variables, where nan values are ignored:")
        for idx, mean in enumerate(means):
            print("feature {idx}: mean={m:.3f}     std={s:.3f}     median={me:.3f}     min={mi:.3f}     max={ma:.3f}.".format(
                  idx=idx, m=mean, s=stds[idx], me=medians[idx], mi=mins[idx], ma=maxs[idx]))
        print()
    return means, stds, medians, mins, maxs



"""Data preparation"""

def find_percentage_of_invalid(train_x, mins, print_vals = True):
    """
    Finds the percentage of invalid values (i.e. -999.0)    
    Input:
    - train_x (ndarray) : input set containing all the features of the data points
    - mins (ndarray) : contains the minimum of each features in the dataset
    - print_vals (boolean) : by default to True, prints the percentage of the invalid values else just return it
    Output:    
    - percent_invalid (dict) : dictinonary that contains the percentage of invalid features for each feature
    """
    percent_invalid = {}
    # iterates on the features
    for idx in range(train_x.shape[1]):
        col = train_x[:,idx]
        # checks that there are indeed invalid values in the feature
        if(mins[idx] == -999.0):
            n = len(col[col==-999])
            p = n*100/len(train_x)
            if(print_vals):
                print("Number of invalid values in feature {idx}: {n} ({p:.3f}%)".format(
                        idx=idx, n=n, p=p))
            # stores the value of the percentage of invalid values
            percent_invalid[idx] = p
    return percent_invalid


def replace_invalids(data, threshold = 50, print_vals = False, med=True):
    """
    Assemble the 3 label vectors with the original ordering    
    Input:
    - data (ndarray) : input set containing the features of the data points
    - threshold (int) : by default to 50, if the number of invalid values is above the threshold in a column, the feature is                               removed if it is under the invalid values are simply replaced by the median or mean
    - print_vals (boolean) : by default to False, if True prints the number of invalid values,of nan values and the new shape                                  after removing them
    - med (boolean) : by default set to True, if True replaces the invalid values by the median else replaces by the mean of the                         feature
    Output:    
    - data (ndarray) : input set without the invalid values
    """
    
    # prints a description of the features if print_vals True
    means, stds, medians, mins, maxs = summarize_features(data, print_vals=print_vals)
    
    if(print_vals):
        print("Number of invalid values: " + str(len(data[data == -999.0])))
        print("Number of Nan values: " + str(np.count_nonzero(np.isnan(data))))
        print("Shape: " + str(data.shape))
        print()
    
    # gets the number of invalid values for each feature
    percent_invalid = find_percentage_of_invalid(data, mins, print_vals=print_vals)
    
    # stores the indices of the features to delete because the number of invalid values is above the threshold
    to_delete = [k for k,v in percent_invalid.items() if v > threshold]
    
    # stores the indices of the features where the invalid values need to be replaces
    change_to_mean = [k for k in percent_invalid.keys() if k not in to_delete]
    
    for idx in change_to_mean:
        # puts the value np.NaN in place of the invalid values in the features to modify to make the measurements on the features
        # without the invalid values
        data[:,idx] = np.where(data[:,idx]==-999, np.NaN, data[:,idx])
        
    
    # calculates the means, medians and means without the invalid values and NaNs
    n_means, _, n_medians, n_mins, _ = summarize_features(data, include_nan=False, print_vals=print_vals)
    
    
    if med:
        for idx in change_to_mean:
            # replacing the invalid values by the median if med is True
            data[:,idx] = np.where(np.isnan(data[:,idx]), n_medians[idx], data[:,idx])
    else :
        for idx in change_to_mean:
            # replacing the invalid values by the mean if med is False
            data[:,idx] = np.where(np.isnan(data[:,idx]), n_means[idx], data[:,idx])

    
    n_percent_invalid = find_percentage_of_invalid(data, n_mins, print_vals=print_vals)
    
    # deletes the features to discard
    data = np.delete(data,to_delete,axis=1)
    
    if(print_vals):
        print()
        print("Number of invalid values: " + str(len(data[data == -999.0])))
        print("Number of Nan values: " + str(np.count_nonzero(np.isnan(data))))
        print("New shape: " + str(data.shape))
    
    return data


def remove_redundants(tr_x, tr_y, threshold = 0.95):
    """
    Assemble the 3 label vectors with the original ordering    
    Input:
    - tr_x (ndarray) : binary prediction for set 1
    - tr_y (ndarray) : binary prediction for set 2
    - threshold (float) : indices of the data points in set 3   
    Output:    
    - tr_x_removed (ndarray) : predicted labels for test set ( with the original ordering)
    - idx (ndarray) :
    """    
    corrm = np.corrcoef(np.hstack([tr_x,tr_y.reshape((-1,1))]).T)
    rows, cols = np.where((corrm>threshold) & (corrm<1.0))
    idx = [c for r,c in zip(rows,cols) if (c>r)]
    tr_x_removed = np.delete(tr_x, idx, axis=1)
    return tr_x_removed, idx



def standardize_cols(x, mean_x=None, std_x=None):
    """
    Assemble the 3 label vectors with the original ordering    
    Inputs:
    - x (ndarray) : dataset to standardize
    - mean_x (ndarray) : by default to None, if an array is passed as an argument, standardize the array using those means
    - std_x (ndarray) : by default to None, if an array is passed as an argument, standardize the array using those standard 
                        deviations
    Outputs:    
    - x (ndarray) : dataset standardized
    - mean_x (ndarray) : means of the features pre-standardization
    - std_x (ndarray) : standard deviatinos of the features pre-standardization
    """
    
    # checks if mean_x is None and if it is, calculates it as well a std_x
    if np.logical_not(np.all(mean_x)):
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0)
    # standardizes
    x = x - mean_x
    x = x / std_x
    return x, mean_x, std_x


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    """
    Assemble the 3 label vectors with the original ordering    
    Input:
    - x (ndarray) : binary prediction for set 1
    - y (ndarray) : binary prediction for set 2
    - ratio (ndarray) : binary prediction for set 3
    - seed (float) : indices of the data points in set 1   
    Output:    
    - train_x (ndarray) : binary prediction for set 1
    - train_y (ndarray) : binary prediction for set 2
    - test_x (ndarray) : binary prediction for set 3
    - test_y (ndarray) : indices of the data points in set 1
    """
    # set seed and shuffle the indices
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
   
    #splits the set according to the ratio on the shuffled set
    ratio_idx = int(np.floor(ratio*len(y)))
    train_y = shuffled_y[:ratio_idx]
    train_x = shuffled_x[:ratio_idx]
    test_y = shuffled_y[ratio_idx:]
    test_x = shuffled_x[ratio_idx:]
    return train_x, train_y, test_x, test_y    


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    only works if x has more than one dimension
    """
    """
    Assemble the 3 label vectors with the original ordering    
    Inputs:
    - x (ndarray) : binary prediction for set 1
    - degree (int) : binary prediction for set 2 
    Outputs:    
    - p (ndarray) : predicted labels for test set ( with the original ordering)
    """
    # forming a matrix containing the data points
    terms = np.hstack([np.ones([x.shape[0],1]),np.tile(x,(1,degree))])
    index = np.arange(degree)+1
    
    # forming a matrix contnaining the exponents
    exponents = np.multiply(np.ones((1, x.shape[1])), index[:, np.newaxis])
    exponents = exponents.reshape([1, x.shape[1]*degree])
    exponents = np.multiply(exponents, np.ones([x.shape[0], 1]))
    exponents = np.hstack([np.ones( (x.shape[0], 1) ),exponents])
    
    # using the exponent matrix as the element-wise exponents of the terms in the terms matrix
    p=np.power(terms,exponents)
    return p


""" Splitting the data in 3 sub-datasets"""

def split_data_jetnum(x, y, n = 22):
    """
    Split the dataset into three datasets in function of the categorical feature PRI_jet_num
    The conditions to split the data-set are:
    PRI jet num = 0
    PRI jet num ≤ 1
    PRI jet num >1  
    Inputs:
    - x (ndarray) : input training data
    - y (ndarray): input training binary labels
    - n (int) : index of feature PRI_jet_num by default 22
    Outputs of the form : 
    - x_i (ndarray): input training data of set i
    - y_i (ndarray): input training binary labels of set i
    """
    indices_1 = np.where(x[:,n] == 0)
    indices_2 = np.where(x[:,n] == 1)
    indices_3 = np.where(x[:,n] > 1)
    
    # delete the PRI_jet_num column
    x = np.delete(x, n, axis = 1)
    
    # split the data
    x_1, y_1 = x[indices_1], y[indices_1]
    x_2, y_2 = x[indices_2], y[indices_2]
    x_3, y_3 = x[indices_3], y[indices_3]
    return x_1, y_1, x_2, y_2, x_3, y_3, indices_1, indices_2, indices_3
    
    
def unsplit_data_jetnum(y_1, y_2, y_3, indices_1, indices_2, indices_3):
    """
    Assemble the 3 label vectors with the original ordering    
    Input:
    - y_1 (ndarray) : binary prediction for set 1
    - y_2 (ndarray) : binary prediction for set 2
    - y_3 (ndarray) : binary prediction for set 3
    - indices_1 (ndarray) : indices of the data points in set 1
    - indices_2 (ndarray) : indices of the data points in set 2
    - indices_3 (ndarray) : indices of the data points in set 3   
    Which obtained with split_data_jetnum()   
    Output:    
    - y (ndarray) : predicted labels for test set ( with the original ordering)
    """
    # assemble the split labels
    y = np.concatenate((y_1, y_2, y_3))
    # assemble the indices that were initally used to split the dataset
    indices = np.concatenate((indices_1[0], indices_2[0], indices_3[0]))
    # sort the labels as they should have been if the dataset was not split
    y = y[indices.argsort()]
    return y