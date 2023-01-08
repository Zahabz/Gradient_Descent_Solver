#!/usr/bin/python3i

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.linear_model import LinearRegression


"""
This python script computes the parameters for a Linear Regression model with one variable using the Gradient Descent Algorithm

"""

# Initialization of random parameters for w and b

np.random.seed(78)

X_val = np.random.randn(18, 6)
Y = np.arange(30, 48)
w_init = np.array([0.5, 7, 3, 0.89, 0.2, 10])
b_init = 7

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
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w_ini) + b_ini
        err = f_wb_i - y[i]
        cost = cost + err ** 2
    cost = cost / ( 2 * m)

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
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0


    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        
        dj_db = dj_db + err

    dj_dw = dj_dw / m
    dj_db = dj_db / m

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
        alpha (float): The learning rate required for gradient descent
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

    lm = LinearRegression()
    reg = lm.fit(X, y)
    Y_hat = lm.predict(X)
    w = reg.coef_
    b = reg.intercept_

    return w, b

print(grad_descent_solver(X_val, Y, w_init, b_init, grad_computation, cost_computation, 10000, 0.4))
print(linear_regression(X_val, Y))
print(X_val)
