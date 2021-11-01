import numpy as np
from plots import *
from proj1_helpers import *
from implementations import *
from grid_search import *
from cross_validation import *
from eda_preprocessing import *
    
    
def split_grid_search_predict_rr(train_x, train_y, test_x, test_y, type_="rr", rmv_redundants=True, hyperparams=None):
    '''
    This function performs grid search on the degree of the polynomial expansion of ridge regression and on the regularization.       term  and trains the model and returns the prediction on the test set. It can also just train a model and return the               prediction if we specify the hyperparameters when calling the function.
    This method was adapted for grid search on ridge regression, but can be modified to grid search any model's hyperparameters.
    Inputs:
    - train_x (ndarray) : train features
    - train_y (ndarray) : train labels
    - test_x (ndarray) : test features
    - test_y (ndarray) : test labels
    - type (string) : model used in this method by default rr for ridge regression
    - rmv_redundants (boolean) : to precise whether we remove redundant features from the training set by default True
    - hyperparms (List) : contains the hyperparameters in the case we don't want to perform grid-search and just predict
                        the labels with these hyperparameters. By default to None  
    Outputs:
    if hyperparms = None :
    - test_preds(ndarray) : contains the predicted labels for the test set with the hyperparameter that has the best accuracy
    - acc_mean_1 (ndarray) : contains all the means of the accuracies for each value of hyperparameter for the first set 
    - acc_std_1 (ndarray) : contains all the standard deviations of the accuracies for each value of hyperparameter for the first 
                            set
    - acc_mean_2 (ndarray) : contains all the means of the accuracies for each value of hyperparameter for the second set
    - acc_std_2 (ndarray) : contains all the standard deviations of the accuracies for each value of hyperparameter for the second 
                            set 
    - acc_mean_3(ndarray) : contains all the means of the accuracies for each value of hyperparameter for the third set
    - acc_std_3 (ndarray) : contains all the standard deviations of the accuracies for each value of hyperparameter for the third 
                            set
    else :
    - test_preds(ndarray) : contains the predicted labels for the test set with the hyperparameter specified in the function   
                            arguments
    - hyperparams(ndarray) : array containing the hyperparameters used for the model
    '''
    # splitting the training set into 3 according to the value of the feature PRI jet_num
    tr_x_1, tr_y_1, tr_x_2, tr_y_2, tr_x_3, tr_y_3, _, _, _ = split_data_jetnum(train_x,train_y)
    
    ### TRAINING SECTION : 
    
    # data pre-processing
    print("Replacing invalids")
    tr_x_1 = replace_invalids(tr_x_1)
    tr_x_1 = np.delete(tr_x_1, -1, axis=1)  # last column is always equal to zero
                                            # so we remove it
    tr_x_2 = replace_invalids(tr_x_2)
    tr_x_3 = replace_invalids(tr_x_3)
    
    print("tr_x_1.shape " + str(tr_x_1.shape))
    print("tr_x_2.shape " + str(tr_x_2.shape))
    print("tr_x_3.shape " + str(tr_x_3.shape))
   
    if(rmv_redundants):
        #handling redundant features
        print("Handling redundant features")
        tr_x_1, idx_1 = remove_redundants(tr_x_1, tr_y_1)
        tr_x_2, idx_2 = remove_redundants(tr_x_2, tr_y_2)
        tr_x_3, idx_3 = remove_redundants(tr_x_3, tr_y_3)

    print("idx_1" + str(idx_1))
    print("idx_2" + str(idx_2))
    print("idx_3" + str(idx_3))
   
    print("tr_x_1.shape after remove redundants" + str(tr_x_1.shape))
    print("tr_x_2.shape after remove redundants" + str(tr_x_2.shape))
    print("tr_x_3.shape after remove redundants" + str(tr_x_3.shape))
   
    
    # standardizing the columns
    print("Standardizing features")
    tr_x_norm_1, means_1, std_1 = standardize_cols(tr_x_1)
    tr_x_norm_2, means_2, std_2 = standardize_cols(tr_x_2)
    tr_x_norm_3, means_3, std_3 = standardize_cols(tr_x_3)
    
    print("means_1.shape " + str(means_1.shape))
    print("means_2.shape " + str(means_2.shape))
    print("means_3.shape " + str(means_3.shape))
    
    
    if(hyperparams is None):     
        deg_list = generate_degree(12)
        lambda_list =  np.concatenate(generate_lambda(6)),(np.array([0]))
        print(deg_list)
        print(lambda_list)      
        print("Selection of hyperparameters for first model")
        # finding the best combination of degree and lambda_ for SET 1
        acc_mean_1, acc_std_1 = grid_search_2_param(tr_y_1, tr_x_norm_1, lambda_list, deg_list, type_= type_)
        acc_1, lambda_1, degree_1 = get_best_parameters_2_param(lambda_list, deg_list, acc_mean_1)
        print("Acc 1: " + str(acc_1))
    else:
        # getting the hyperparameters from the function argument
        degree_1 = hyperparams[0][0]
        lambda_1 = hyperparams[1][0]
        
    print("Degree 1: " + str(degree_1))
    print("Lambda 1: " + str(lambda_1))
    
    print("Training first model")
    
    tx_norm_p_1 = build_poly(tr_x_norm_1, degree_1)
    tr_rr_w1, tr_rr_loss1 = ridge_regression(tr_y_1, tx_norm_p_1, lambda_1)
    
  
    if(hyperparams is None):      
        print("Selection of hyperparameters for second model")
        # finding the best combination of lambda_ and degree for SET 2
        acc_mean_2, acc_std_2 = grid_search_2_param(tr_y_2, tr_x_norm_2, lambda_list, deg_list, type_= type_)
        acc_2, lambda_2, degree_2 = get_best_parameters_2_param(lambda_list, deg_list, acc_mean_2)
        print("Acc 2 " + str(acc_2))
    else:  
        # getting the hyperparameters from the function arguments
        degree_2 = hyperparams[0][1]
        lambda_2 = hyperparams[1][1]
        
    print("Degree 2 " + str(degree_2))
    print("Lambda 2 " + str(lambda_2))

    
    print("Training second model")
    
    tx_norm_p_2 = build_poly(tr_x_norm_2, degree_2)
    tr_rr_w2, tr_rr_loss2 = ridge_regression(tr_y_2, tx_norm_p_2, lambda_2)
    
    
    
    if(hyperparams is None):      
        print("Selection of hyperparameters for third model")
        # finding the best combination of lambda_ and degree for SET 3
        acc_mean_3, acc_std_3 = grid_search_2_param(tr_y_3, tr_x_norm_3, lambda_list, deg_list, type_= type_)
        acc_3, lambda_3, degree_3 = get_best_parameters_2_param(lambda_list, deg_list, acc_mean_3)
        print("Acc 3 " + str(acc_3))
    else:
        # getting the hyperparameters from the function arguments
        degree_3 = hyperparams[0][2]
        lambda_3 = hyperparams[1][2]
    
    print("Training third model")
    tx_norm_p_3 = build_poly(tr_x_norm_3, degree_3)
    tr_rr_w3, tr_rr_loss3 = ridge_regression(tr_y_3, tx_norm_p_3, lambda_3)
    print("Degree 3 " + str(degree_3))
    print("Lambda 3 " + str(lambda_3))
    
    
    ## PREDICTION SECTION : 
    
    print("Prediction of test set")
    te_x_1, te_y_1, te_x_2, te_y_2, te_x_3, te_y_3, te_indexes_1, te_indexes_2, te_indexes_3 = split_data_jetnum(test_x,test_y)
    
    # preprocessing on test set
    te_x_1 = replace_invalids(te_x_1)
    te_x_1 = np.delete(te_x_1, -1, axis=1) 
    te_x_2 = replace_invalids(te_x_2)
    te_x_3 = replace_invalids(te_x_3)
    
    print("te_x_1.shape before remove redundants" + str(te_x_1.shape))
    print("te_x_2.shape before remove redundants" + str(te_x_2.shape))
    print("te_x_3.shape before remove redundants" + str(te_x_3.shape))
    
    if(rmv_redundants):
        # removing the same redundant features as in the training set
        te_x_1 = np.delete(te_x_1, idx_1, axis=1)
        te_x_2 = np.delete(te_x_2, idx_2, axis=1)
        te_x_3 = np.delete(te_x_3, idx_3, axis=1) 
        print("te_x_1.shape after remove redundants" + str(te_x_1.shape))
        print("te_x_2.shape after remove redundants" + str(te_x_2.shape))
        print("te_x_3.shape after remove redundants" + str(te_x_3.shape))

    # standardize the test set with the train statistics
    te_x_norm_1, _, _ = standardize_cols(te_x_1, means_1, std_1)
    te_x_norm_2, _, _ = standardize_cols(te_x_2, means_2, std_2)
    te_x_norm_3, _, _ = standardize_cols(te_x_3, means_3, std_3)

    # polynomial expansion

    te_x_norm_p_1 = build_poly(te_x_norm_1, degree_1)
    te_x_norm_p_2 = build_poly(te_x_norm_2, degree_2)
    te_x_norm_p_3 = build_poly(te_x_norm_3, degree_3)
    
    #final prediction
    te_y_1_eval = predict_labels(tr_rr_w1, te_x_norm_p_1)
    te_y_2_eval = predict_labels(tr_rr_w2, te_x_norm_p_2)
    te_y_3_eval = predict_labels(tr_rr_w3, te_x_norm_p_3)
    
    #merging all the test prediction in one array
    test_preds = unsplit_data_jetnum(te_y_1_eval, te_y_2_eval, te_y_3_eval, te_indexes_1, te_indexes_2, te_indexes_3)
    
    if(hyperparams is None):
        return test_preds, acc_mean_1, acc_std_1, acc_mean_2, acc_std_2, acc_mean_3, acc_std_3
    else:
        return test_preds, hyperparams