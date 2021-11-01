# Useful starting lines
import numpy as np
from proj1_helpers import *

"""Functions used to compute the mse loss, gradient and gradient descent."""

def compute_loss_mse(y, tx, w, mae = False):
    """
    Computes MSE or MAE loss given input data, weights and labels.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights to evaluate
    - mae (bool): returns MAE instead if MSE if True, default to False
    Outputs: 
    - mse (float): mean square error of the predictions made by w
    """
    e = y - tx.dot(w)
    if(mae):
        return np.mean(np.abs(e))
    return 1/2*np.mean(e**2)

def compute_loss_rmse(y, tx, w):
    """
    Computes RMSE loss using MSE given input data, weights and labels.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights to evaluate
    Outputs: 
    - rmse (float): root mean square error of the predictions made by w
    """
    return np.sqrt(2*compute_loss_mse(y, tx, w))
    

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Gradient Descent"""

def compute_gradient_mse(y, tx, w):
    """
    Computes the gradient of the MSE loss given input data, weights and labels.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights
    Outputs: 
    - grad (ndarray): gradient of the mse loss corresponding to the given y, x, and w
    - e (ndarray): error vector corresponding to distance between predictions and actual labels
    """
    e = y - np.dot(tx, w)
    grad = np.dot(tx.T, e)/(-len(e))
    return grad, e


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Computes the gradient descent using MSE loss given input parameter.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - initial_w (ndarray): inital weights vector (usually only zeros).
    - max_iters (int): number of iterations of gradient descent to perform during training
    - gamma (float): learning rate for weights updtate at each iteration of the gradient descent
    Outputs: 
    - ws (list): list of weights updated at each iteration. ws[-1] are the final weights. 
    - losses (list): list of losses obtained at each iteration
    """
    # Define parameters to store w and loss
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad,_ = compute_gradient_mse(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        # update weights
        w = w - gamma*grad
        losses.append(w)
    return w, losses[-1]


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Stochastic Gradient Descent"""

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - batch_size (int): size of the subsets of matching elements from y and tx. 
    - num_batches (int): number of subsets of matching elements from y and tx. Default to 1
    - shuffle (bool): data will be shuffled before being split if True. Default to True
    Outputs: 
    - minibatch_y (iterator): iterator which gives mini-batches of batch-size from y
    - minibatch_tx (iterator): iterator which gives mini-batches of batch-size from tx
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            

def compute_stoch_gradient_mse(y, tx, w):
    """
    Computes the gradient of the MSE loss given input data, weights and labels.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights
    Outputs: 
    - grad (ndarray): gradient of the mse loss corresponding to the given y, x, and w
    - e (ndarray): error vector corresponding to distance between predictions and actual labels
    """
    return compute_gradient_mse(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Performs the stochastic gradient descent algorithm.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - initial_w (ndarray): inital weights vector (usually all zeros)
    - max_iters (int): number of iterations of SGD to perform during training
    - gamma (float): learning rate for weights updtate at each iteration of the SGD
    Outputs: 
    - ws (list): list of weights updated at each iteration. ws[-1] are the final weights. 
    - losses (list): list of losses obtained at each iteration
    """
    batch_size = 1
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            # compute grad on this minibatch
            grad, _ = compute_stoch_gradient_mse(minibatch_y, minibatch_tx, w)
            # update w
            w = w - gamma*grad
            loss = compute_loss_mse(minibatch_y, minibatch_tx, w)
            losses.append(loss)
    return w ,losses[-1]



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Least Squares using Normal Equations"""

def least_squares(y, tx):
    """
    Calculate the least squares normal equations solution.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    Outputs: 
    - w_star (ndarray): optimal set of weights according to the least squares normal equations
    - losses (float): mse loss computed with w_star on tx and y
    """
    w_star = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_loss_mse(y, tx, w_star)
    return w_star, loss



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Ridge regression"""


def build_model_data(x, y):
    """
    Form (y,tX) to get regression data in matrix form.
    Inputs:
    - x (ndarray): training data
    - y (ndarray): training desired values
    Outputs: 
    - y (ndarray): training desired values
    - tx (ndarray): training data with additional column of ones (for the bias) at the start
    """
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
    
    
def ridge_regression(y, tx, lambda_):
    """
    Implement ridge regression with the given penality hyperparameter lambda
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - lambda_ (float): regularization term
    Outputs: 
    - w (ndarray): optimal set of weights according to the ridge regression algorithm
    - losses (float): rmse loss computed with w on tx and y
    """
    a=tx.T.dot(tx)
    a=a+lambda_*np.eye(len(a))*2*len(tx)
    b=tx.T.dot(y)
    
    w=np.linalg.solve(a,b)
    loss = compute_loss_rmse(y, tx, w)
    return w, loss


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Logistic regression"""

def sigmoid(t):
    """
    Apply the sigmoid function on t
    Inputs:
    - t (ndarray): vector on which to apply the sigmoid
    Outputs: 
    - sigmoid(t) (ndarray): output of the sigmoid function
    """
    return 1.0/(1+np.exp(-t))


def compute_loss_neg_log(y, tx, w):
    """
    Compute the negative log likelihood loss given y, tx and weights
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights
    Outputs: 
    - loss (float): negative log likelihood loss computed with w on tx and y
    """
    pred_y = sigmoid(tx.dot(w))
    l = y.T.dot(np.log(pred_y+1e-5)) + (1-y).T.dot(np.log(1-pred_y+1e-5))
    return np.squeeze(-l)


def compute_gradient_neg_log(y, tx, w):
    """
    Compute the gradient of the negative log likelihood loss given y, tx and weights
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights
    Outputs: 
    - gradient (float): gradient of the negative log likelihood loss computed with w on tx and y
    """
    pred = sigmoid(tx.dot(w))
    e = (pred - y).reshape(-1,1)

    return np.dot(tx.T, e)



def learning_by_gradient_descent_neg_log(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights
    - gamma (float): learning rate
    Outputs: 
    - w (ndarray): updated weights
    - loss (float): negative log likelihood loss computed with w on tx and y
    """
    loss = compute_loss_neg_log(y, tx, w)
    grad = compute_gradient_neg_log(y, tx, w)
    w = w - gamma*np.squeeze(grad)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Train the logistic regression model.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - initial_w (ndarray): inital weights vector (usually all zeros)
    - max_iters (int): number of iterations to perform during training
    - gamma (float): learning rate
    Outputs: 
    - w (ndarray): last set of weights obtained through training
    - losses (list): list of losses computed during training
    """
    # init parameters
    max_iter = max_iters
    threshold = 1e-8
    losses = []
    w = initial_w
    
    if tx[:,0].sum() != tx.shape[0]:
        #add zero before beginning of weights
        w = np.insert(w, 0, 0).reshape(-1, 1)
        tx = np.c_[np.ones((y.shape[0], 1)), tx]
    
    # start the logistic regression
    for it in range(max_iter):
        # get loss and update w.
        w, loss = learning_by_gradient_descent_neg_log(y, tx, w, gamma)
        # save loss
        losses.append(loss)
        # if the step had too little effect, stop the training: training converged
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Penalized logistic regression"""


def calculate_hessian_neg_log(y, tx, w):
    """
    Compute the Hessian matrix of the negative log likelihood loss given y, tx and weights
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights
    Outputs: 
    - hessian (ndarray): hessian of the negative log likelihood loss computed with w on tx and y
    """
    pred = sigmoid(np.dot(tx, w)).reshape((-1,1))
    s = np.multiply(pred, 1-pred)
    # element-by-element multiplication instead of matrix multpication with diag matrix
    dot = np.multiply(s, tx)
    return np.dot(tx.T, dot)


def penalized_logistic_regression_neg_log(y, tx, w, lambda_):
    """
    Return the penalized loss, gradient and hessian of the negative log likelihood given y, tx and weights
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights
    - lambda_ (float): penalization term
    Outputs: 
    - loss (float): penalized negative log likelihood loss
    - gradient (ndarray): penalized gradient of the negative log likelihood loss
    - hessian (ndarray): penalized hessian of the negative log likelihood loss
    """
    loss = compute_loss_neg_log(y, tx, w) + lambda_*np.squeeze((w.T.dot(w)))
    grad = compute_gradient_neg_log(y, tx, w) + 2*lambda_*w.reshape(-1, 1)
    hessian = calculate_hessian_neg_log(y, tx, w) + 2
    return loss, grad, hessian


def learning_by_penalized_gradient_neg_log(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent using the penalized logistic regression.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - w (ndarray): weights
    - gamma (float): learning rate
    - lambda_ (float): penalization term
    Outputs: 
    - w (ndarray): updated weights
    - loss (float): penalized negative log likelihood loss computed with w on tx and y
    """
    loss, gradient, hessian = penalized_logistic_regression_neg_log(y, tx, w, lambda_)
    w -= gamma * np.squeeze(np.linalg.solve(hessian, gradient))
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Train the regularized logistic regression model.
    Inputs:
    - y (ndarray): training desired values
    - tx (ndarray): training data
    - lambda_ (float): penalization term
    - initial_w (ndarray): inital weights vector (usually all zeros)
    - max_iters (int): number of iterations to perform during training
    - gamma (float): learning rate
    Outputs: 
    - w (ndarray): last set of weights obtained during training
    - losses (list): list of losses computed during training
    """
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    
    if tx[:,0].sum() != tx.shape[0]:
        w = np.insert(w, 0, 0).reshape(-1, 1)
        tx = np.c_[np.ones((y.shape[0], 1)), tx]
    
    # start the logistic regression
    for it in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_penalized_gradient_neg_log(y, tx, w, gamma, lambda_)
        # save loss
        losses.append(loss)
        # if the step had too little effect, stop the training: training converged
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]


"""Accuracy measurements"""

def num_correct_pred(y_pred,y):
    """
    Calculate the number of correct predictions
    Inputs:
    - y_pred (ndarray): predicted values
    - y (ndarray): training desired values
    Outputs: 
    - n (int): number of correct predictions
    """
    return np.sum(np.equal(y_pred,y))

def accuracy(y, x, weights):
    """
    Calculate the percentage of correct predictions
    Inputs:
    - y (ndarray): training desired values
    - x (ndarray): input training values
    - weights (ndarray): final weights for prediction
    Outputs: 
    - accuracy (float): prediction accuracy of the given weights
    """
    y_pred = np.squeeze(predict_labels(weights, x))
    return 100*np.sum(np.equal(y_pred,y))/len(y)



