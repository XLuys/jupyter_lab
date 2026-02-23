import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import t
from sklearn import datasets, linear_model
from scipy import stats
from scipy.linalg import sqrtm
from numpy.linalg import inv
from IPython.display import display

eps = 0.01
alpha = 0.1
steps = 1000

def var(y_orig, y_pred):
    cost = 0
    cost += ((y_orig - y_pred) ** 2)
    cost = cost.sum()
    return cost

def mse(y_orig, y_pred):
    return var(y_orig, y_pred) / y_orig.shape[0]

def RSS(y_orig, y_pred, df):
    return var(y_orig, y_pred) / (y_orig.shape[0] - df)

def R_sq(y_orig, y_pred):
    n = y_orig.shape[0]
    return (y_orig.var()*(n) - var(y_orig, y_pred)) / (y_orig.var()*(n))

def R_sq_adj(y_orig, y_pred, df):
    n = y_orig.shape[0]
    return 1 - (1 - R_sq(y_orig, y_pred)) * (n-1)/(n - df)

def RSE(y, y_pred, df):
    n = len(y)
    return math.sqrt(var(y, y_pred)/(n - df))

def F_stat(y, y_pred, df):
    p = df - 1
    n = len(y)
    TSS = y.var() * (n-1)
    RSS = var(y, y_pred)
    return ((TSS - RSS) / p)/(RSS / (n - p - 1))

def gradient_descent(X, y, W, int_, alpha):
    n = X.shape[1]
    y_pred = np.sum(np.multiply(W, X), axis = 0) + int_
    # y_pred = np.sum(np.multiply(W, X)) + int_
    error = y_pred - y
    
    W_d = (1/n) * np.dot(X, error)[:, None]
    int_d = (1/n) * np.sum(error)
    
    new_W = W - alpha * W_d
    new_int_ = int_ - np.sum(alpha * int_d)

    return new_W, new_int_

def my_fit(X, y):
    X_rows = X.T
    int_ = 0
    W = np.zeros((len(X_rows), 1))
    STD = np.std(X_rows, axis=1, keepdims=True)
    MEAN = np.mean(X_rows, axis=1, keepdims=True)
    X_scaled = np.divide((X_rows - MEAN), STD)

    
    W_history = []
    int_history = []
    
    for step in range(steps):
        W_new, int_new = gradient_descent(X_scaled, y, W, int_, alpha)
        W_history.append(W_new)
        int_history.append(int_new)

        if (abs(W_new - W)<eps).any() and abs(int_new - int_) < eps:
            break
            
        W, int_ = (W_new, int_new)
    W_orig = np.divide(W, STD).flatten()
    int_orig = int_ - np.divide(np.multiply(W, MEAN), STD).sum()
    
    return (W_orig, int_orig, W_history, int_history)

def get_prediction(X, coef_):
    return np.sum(np.multiply(coef_, X), axis = 0)[0]

def get_regression_stats(X, y, int_=True):
    df = X.shape[1] + 1
    if int_ is True:
        X2 = np.insert(X, 0, 1, axis=1)
    else:
        X2 = X
    X_T = X2.T
    X_inv = inv(np.matmul(X2.T, X2))

    B = np.matmul(np.matmul(X_inv, X_T), y)[:, None]
    y_pred = np.sum(np.multiply(B, X_T), axis = 0)
    
    stats_df = pd.DataFrame()
    meta_stats_df = pd.DataFrame()
    
    names = ['RSE', 'R^2', 'Adj R^2', 'F-value']
    meta_stats_df['names'] = names
    meta_stats_df = meta_stats_df.set_index('names')
    
    names = [f"b{i}" for i in range(B.shape[0])]
    stats_df['names'] = names
    stats_df = stats_df.set_index('names')
    stats_df['coef'] = B
    ### Coefficient Statistics
    
    STD = RSS(y, y_pred, df)
    rse = RSE(y, y_pred, df)
    r_sq = R_sq(y, y_pred)
    r_sq_adj = R_sq_adj(y, y_pred, df)
    cov_mat = STD * X_inv
    SE = np.sqrt(np.diag(cov_mat))[:, None]
    stats_df['Std. err.'] = SE.flatten()

    T = np.divide(B, SE)
    stats_df['t-stat'] = T.flatten()

    F = F_stat(y, y_pred, df)
    
    P = 2 * t.sf(np.abs(T), X.shape[0]-df)
    stats_df['p-value'] = P.flatten()

    conf_int_low = []
    conf_int_high = []
    for i in range(len(B)):
        low = B[i] - 1.97 * SE[i]
        high = B[i] + 1.97 * SE[i]
        conf_int_low.append(low[0])
        conf_int_high.append(high[0])
    stats_df['2.5%'] = conf_int_low
    stats_df['97.5%'] = conf_int_high

    ### Meta-Statistics
    meta_values = [rse, r_sq, r_sq_adj, F]
    meta_stats_df['values'] = meta_values

    return stats_df, meta_stats_df