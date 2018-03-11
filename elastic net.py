# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 18:57:04 2018

@author: Riad Ghorra
"""

import scipy.io
import numpy as np
from sklearn import linear_model
from scipy import linalg
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
# Exercises for lecture 3 in 02582, lars

# Helper functions for data handling
def center(X):
    """ Center the columns (variables) of a data matrix to zero mean.
        
        X, MU = center(X) centers the observations of a data matrix such that each variable
        (column) has zero mean and also returns a vector MU of mean values for each variable.
     """ 
    n = X.shape[0]
    mu = np.mean(X,0)
    X = X - np.ones((n,1)) * mu
    return X, mu

def normalize(X):
    """Normalize the columns (variables) of a data matrix to unit Euclidean length.
    X, MU, D = normalize(X)
    i) centers and scales the observations of a data matrix such
    that each variable (column) has unit Euclidean length. For a normalized matrix X,
    X'*X is equivalent to the correlation matrix of X.
    ii) returns a vector MU of mean values for each variable.
    iii) returns a vector D containing the Euclidean lengths for each original variable.
    
    See also CENTER
    """
    
    n = np.size(X, 0)
    X, mu = center(X)
    d = np.linalg.norm(X, ord = 2, axis = 0)
    d[np.where(d==0)] = 1
    X = np.divide(X, np.ones((n,1)) * d)
    return X, mu, d

def normalizetest(X,mx,d):
    """Normalize the observations of a test data matrix given the mean mx and variance varx of the training.
       X = normalizetest(X,mx,varx) centers and scales the observations of a data matrix such that each variable
       (column) has unit length.
       Returns X: the normalized test data"""
    
    n = X.shape[0]
    X = np.divide(np.subtract(X, np.ones((n,1))*mx), np.ones((n,1)) * d)
    return X

data = pd.read_csv('Case1_Data-bis.csv')

training = data.loc[data['Y'].notnull()]
prediction = data.loc[data['Y'].isnull()]
train_y = training['Y'].as_matrix()
train_X = training[[col for col in training.columns if col != 'Y']].as_matrix()

# data preprocessing

def oneOutOfK(data, col_num, removeOriginal=True):
    '''Takes the column specified and created one 
    out of K binary columns for the values in that
    column. If removeOriginal is set to True, the 
    original column will be taken out of the data.
    Returns a numpy array.'''
    [n, p] = data.shape
    values = data[:, col_num]
    new_cols = sorted(list(set(values)))
    for col in new_cols:
        new_col = np.array([1 if row[col_num] == col else 0 for row in data])
        data = np.append(data, new_col.reshape((n, 1)), axis=1)
    if removeOriginal:
        data = np.delete(data, col_num, axis=1)
    return data
   
def missing_value_as_mean(training_data):
    data = training_data
    for i, column in enumerate(data.T[:-1,:]):
            # column = train_X[column].as_matrix()
            column_without_nan = column[~np.isnan(np.array(column, dtype=float))]
            column_mean = column_without_nan.mean()
            # print( column_without_nan.mean() )
            for predictor_index in range(len(column)):
                if pd.isnull(np.array(column[predictor_index], dtype=float)):
                    column[predictor_index] = column_mean

            data.T[i, :] = column
    return data

train_X = oneOutOfK(train_X, 99)
train_X = missing_value_as_mean(train_X)

# split Train / Test (70% / 30%) (for train_X)
test_split_ratio = 0.3

X_train, X_test, y_train , y_test= train_test_split(train_X, train_y, test_size=test_split_ratio, random_state=0)


[n,p] = X_train.shape
lambdas = np.array([1e-6, 1e-3, 1e0, 1e3])
K = range(0,11,1) # Ratio between L1 and L2 norms

CV = 5 # if CV = n leave-one-out, you may try different numbers
# this corresponds to crossvalind in matlab
# permutes observations - useful when n != 0
I = np.asarray([0] * n)
for i in range(n):
    I[i] = (i + 1) % CV + 1
     
I = I[np.random.permutation(n)]
K = range(0,11,1)

Err_tr = np.zeros((CV,len(K)))
Err_tst = np.zeros((CV, len(K))) 
K_elasticNet = np.empty((100, len(lambdas), len(K)))

for i in range(1, CV+1):
    # Split data according to the earlier random permutation
    Ytr = y_train[I != i].ravel() # ravel collapses the array, ie dim(x,1) to (x,)
    Ytst = y_train[I == i].ravel()
    Xtr = X_train[I != i, :]
    Xtst = X_train[I == i, :]

    my = np.mean(Ytr)
    Ytr, my = center(Ytr) # center training response
    Ytr = Ytr[0,:] # Indexing in python thingy, no time to solve it
    Ytst = Ytst-my # use the mean value of the training response to center the test response
    mx =np.mean(Xtr,0)
    varx = np.var(Xtr, 0)
    Xtr=Xtr.astype(float)
    Xtst=Xtst.astype(float)
    Xtr, mx, varx = normalize(Xtr) # normalize training data
    Xtst = normalizetest(Xtst, mx, varx)
           #np.ones((np.size(Xtst, 0), p)) *
    # NOTE: If you normalize outside the CV loop the data implicitly carry information of the test data
    # We should perform CV "the right way" and keep test data unseen.
    for k, _lambda in enumerate(lambdas):
        for j, ratio in enumerate(K):
            # Note that the elastic net in sklearn automatically cycles through all the parameters to find best fit
            reg_elastic = linear_model.ElasticNet(alpha = _lambda, l1_ratio = ratio/10, fit_intercept = False) # L1-ratio, how much ridge or lasso, l1_ratio = 1 is the lasso
            reg_elastic.fit(Xtr, Ytr)

            beta = reg_elastic.coef_.ravel()
           # Betas[i-1, :] = beta
        
        # Predict with this model, and find error
        YhatTr = np.matmul(Xtr, beta)
        YhatTest = np.matmul(Xtst, beta)
        Err_tr[i-1, j] = np.matmul((YhatTr-Ytr).T,(YhatTr-Ytr))/len(Ytr) # training error
        Err_tst[i-1, j] = np.matmul((YhatTest-Ytst).T,(YhatTest-Ytst))/len(Ytst) # test error
        
err_tr = np.mean(Err_tr, axis=0) # mean training error over the CV folds
err_tst = np.mean(Err_tst, axis=0) # mean test error over the CV folds
err_ste = np.std(err_tst, axis=0)/np.sqrt(CV) # Note: we divide with sqrt(n) to get the standard error as opposed to the standard deviation    