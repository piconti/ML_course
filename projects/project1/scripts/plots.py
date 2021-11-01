# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_train_test(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Ridge regression for polynomial degree " + str(degree))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("ridge_regression")
    
    
def correlation_plot(tX,y):
    """
    stack  the label matrix y to the feature matrix tX and 
    calculate the correlation coefficients
    """
    corrm=np.corrcoef(np.hstack([tX,y.reshape((-1,1))]).T)
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(corrm, vmax=1.0, cmap=None, center=0, robust=0, fmt='.2f',
                square=True, linewidths=.1, annot=True, cbar_kws={"shrink": .50},annot_kws={"fontsize":8})
    plt.show();