import numpy as np
from implementations import *
from eda_preprocessing import *
from grid_search import *
from proj1_helpers import *


train_y, train_x, train_id = load_csv_data("Data/train.csv")
test_y, test_x, test_id = load_csv_data("Data/test.csv")

# splitting the training set into 3 according to value PRIjetnum
tr_x_1, tr_y_1, tr_x_2, tr_y_2, tr_x_3, tr_y_3, _, _, _ = split_data_jetnum(train_x,train_y)

#Hyper-parameter that achieve the best prediction
hyperparams = [[9, 12, 12],[1e-5, 1e-3, 1e-4]]

### TRAINING SECTION : 

# data pre-processing
print("Replacing invalids")
tr_x_1 = replace_invalids(tr_x_1)
tr_x_1 = np.delete(tr_x_1, -1, axis=1)  # last column is always equal to zero
                                        # so we remove it
tr_x_2 = replace_invalids(tr_x_2)
tr_x_3 = replace_invalids(tr_x_3)



#handling redundant features
print("Handling redundant features")
tr_x_1, idx_1 = remove_redundants(tr_x_1, tr_y_1)
tr_x_2, idx_2 = remove_redundants(tr_x_2, tr_y_2)
tr_x_3, idx_3 = remove_redundants(tr_x_3, tr_y_3)



# standardizing the columns
print("Standardizing features")
tr_x_norm_1, means_1, std_1 = standardize_cols(tr_x_1)
tr_x_norm_2, means_2, std_2 = standardize_cols(tr_x_2)
tr_x_norm_3, means_3, std_3 = standardize_cols(tr_x_3)




degree_1 = hyperparams[0][0]
lambda_1 = hyperparams[1][0]

print("Degree 1: " + str(degree_1))
print("Lambda 1: " + str(lambda_1))

print("Training first model")

tx_norm_p_1 = build_poly(tr_x_norm_1, degree_1)
tr_rr_w1, tr_rr_loss1 = ridge_regression(tr_y_1, tx_norm_p_1, lambda_1)


degree_2 = hyperparams[0][1]
lambda_2 = hyperparams[1][1]

print("Training second model")

print("Degree 2 " + str(degree_2))
print("Lambda 2 " + str(lambda_2))


tx_norm_p_2 = build_poly(tr_x_norm_2, degree_2)
tr_rr_w2, tr_rr_loss2 = ridge_regression(tr_y_2, tx_norm_p_2, lambda_2)



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



te_x_1 = np.delete(te_x_1, idx_1, axis=1)
te_x_2 = np.delete(te_x_2, idx_2, axis=1)
te_x_3 = np.delete(te_x_3, idx_3, axis=1) 


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

test_preds = unsplit_data_jetnum(te_y_1_eval, te_y_2_eval, te_y_3_eval, te_indexes_1, te_indexes_2, te_indexes_3)

create_csv_submission(test_id, test_preds, "output.csv")