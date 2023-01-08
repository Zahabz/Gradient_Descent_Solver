#!/usr/bin/python3


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import time
from sklearn.linear_model import LinearRegression


"""
This python script computes the parameters for a Linear Regression model with one variable using the Gradient Descent Algorithm

"""

# Initialization of random parameters for w and b

np.random.seed(78)

X_val = np.random.randn(18, 6)
Y = np.arange(30, 48)
w_init = np.zeros(6)
b_init = 7

def z_score_norm(X):
    """
    Returns the X array after Z-score normalization

    Parameters:
    -----------
        X(ndarray): A NumPy array of the training data to be used on the model

    Returns:
    --------
        X_norm
    """
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    X_norm = (X - X_mean) / X_std


    return X_norm



def cost_computation(X, y, w_ini, b_ini):
    """
    Returns the cumulative cost using the initial parameters w and b

    Parameters:
    -----------
        X (ndarray): A NumPy array of the predictor variables
        Y (ndarray): A NumPy array consisting of the target variables
        w_ini (int): Initial value of the `w` parameter
        b_ini (int): Initial value of the `b` parameter

    Returns:
    --------
        The total cost of using the initial predictor variables to fit the model
    """
    X_norm = z_score_norm(X)
    m = X_norm.shape[0]
    f_wb = X_norm @ w_ini + b_ini
    err = f_wb - y

    cost = (1 / (2 * m)) * np.sum(err ** 2)

    return cost

def grad_computation(X, y, w, b):
    """
    Returns the value of the initial gradient of the parameters b and w

    Parameters:
    -----------
        X (ndarray): A NumPy array of the predictor variables
        Y (ndarray): A NumPy array consisting of the target variables
        w_ini (int): Initial value of the `w` parameter
        b_ini (int): Initial value of the `b` parameter

    Returns:
    --------
        dj_dw and dj_db
    """
    X_norm = z_score_norm(X)
    m, n = X_norm.shape
    f_wb = X_norm @ w + b
    e = f_wb - y
    dj_dw = (1/m) * (X_norm.T @ e)
    dj_db = (1/m) * np.sum(e)

    return dj_dw, dj_db

def grad_descent_solver(X, y, w, b, gradient_func, cost_func,  num_iters, alpha):
    """
    Returns the values of w and b that fit the data to the SLR model

    Parameters:
    -----------
        X (ndarray): A NumPy array of the predictor variables
        Y (ndarray): A NumPy array consisting of the target variables
        w_ini (int): Initial value of the `w` parameter
        b_ini (int): Initial value of the `b` parameter
        alpha (float): The learning rate required fRor gradient descent
        gradient_func: Calls gradient_computation function during iterations
        cost_func: Calls cost_computation function
        num_iters (int): Iterations needed for convergence

    Returns:
    --------
        (w_final, b_final) for the SLR MODEL

    """
    nums = []
    cost_vals = []
    for num in range(num_iters):

        dj_dwi, dj_dbi = gradient_func(X, y, w, b)

        w = w - alpha * dj_dwi
        b = b  - alpha * dj_dbi

        cost = cost_func(X, y, w, b)

        if num % 100 == 0:
            nums.append(num)
            cost_vals.append(cost)
            print(f' Iteration {num} : Cost: {cost}')

    fig, ax = plt.subplots(1, 1, figsize=(15,6))
    ax.plot(nums, cost_vals)
    ax.set_xlabel('# Iterations')
    ax.set_ylabel('Cost')
    plt.title('Learning curve')
    plt.show()

    return w, b

def linear_regression(X, y):
    """
    Returns the values of w and b using the scikit-learn library

    Parameters:
    -----------
        X (ndarray): A NumPy array of the predictor variables
        Y (ndarray): A NumPy array consisting of the target variables

    Returns:
    --------
        w, b for the linear regression model
    """
    X_norm = z_score_norm(X)
    lm = LinearRegression()
    reg = lm.fit(X_norm, y)
    Y_hat = lm.predict(X_norm)
    w = reg.coef_
    b = reg.intercept_

    return w, b

def plot_comparison(X, y, w, b, gradsolve_func, cost_func):

    """
    Returns a comparison between the target and predicted variables after linear regression

    Parameters:
    -----------
        X (ndarray): A NumPy array containing the training data
        Y (ndarray): A NumPy array containing the target variables
        w (ndarray): The initial weights before applying the Gradient Descent Algorithm
        b (int): The initial intercept(bias)
        gradsolve_func : Calls the grad_descent_solver function
        cost_func: Calls cost_computation function

    Returns:
    --------
        A matplotlib `Axes` comparing the target and predicted variables.
    """
    w_final, b_final = gradsolve_func(X, Y, w, b, grad_computation, cost_computation, 10000, 0.4)
    X_norm = z_score_norm(X)
    y_norm = X_norm @ w_final + b_final
    fig, ax = plt.subplots(1, 6, figsize=(15, 5), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X[:, i], y, label='target')
        ax[i].set_xlabel(f'X_feature_{i + 1}')
        ax[i].scatter(X[:, i], y_norm, label='predict')

    ax[0].set_ylabel('Y values')
    ax[0].legend()
    fig.suptitle('Target versus Prediction')
    plt.show()

plot_comparison(X_val, Y, w_init, b_init, grad_descent_solver, cost_computation)
