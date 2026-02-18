import numpy as np
import math

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
