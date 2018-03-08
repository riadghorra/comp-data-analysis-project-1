#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:35:03 2018

@author: wisse
"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from preprocessing import oneOutOfK, missing_predictor_as_mean, missing_predictor_as_value, missing_predictor_as_knn
from sklearn.model_selection import KFold
import scipy.linalg as lng



# load and display data
data = pd.read_csv('Case1_Data.csv')
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

# split data in training / prediction based of Y value
training = data.loc[data['Y'].notnull()]
prediction = data.loc[data['Y'].isnull()]

print ('We have', len(training), 'training observations and', 
       len(prediction), 'prediction observations')

train_y = training['Y'].as_matrix()
train_X = training[[col for col in training.columns if col != 'Y']].as_matrix()


# One out of K encoding for categorical value
train_X = oneOutOfK(train_X, 99).astype(float)

# Flag for what handling of missing values to use
KNN_rpl = False


# replace missing values with mean of column
train_X_mn = missing_predictor_as_mean(train_X)

# replace missing values with mean of KNN
clean_data = missing_predictor_as_value(train_X, 0)
train_X_knn = missing_predictor_as_knn(5, train_X, clean_data)

test_split_ratio = 0.3

if (KNN_rpl):
    X_train, X_test, y_train , y_test= train_test_split(train_X_knn, train_y, test_size=test_split_ratio, random_state=0)
else:
    X_train, X_test, y_train , y_test= train_test_split(train_X_mn, train_y, test_size=test_split_ratio, random_state=0)

# split Train / Test (70% / 30%) (for train_X)

# First, we'll establish a baseline for the prediction with OLS 
# on the data with NaN's replaced with the column's mean
 # Linear solver
[n_train, p_train], [n_test, p_test] = np.shape(X_train), np.shape(X_test)

 
off_train, off_test = np.ones(n_train), np.ones(n_test)
M_train, M_test = np.c_[off_train, X_train], np.c_[off_test, X_test] # Include offset / intercept
beta, res, rnk, s = lng.lstsq(M_train, y_train)
yhat = np.matmul(M_test, beta)

#residuals
res = (y_test - yhat) ** 2

print('MSE for OLS with {} replacement of NaN ='.format('knn' if KNN_rpl else 'mean'), np.mean(res))

K = 2

# K - fold cross validation 
kf = KFold(K, shuffle=True)

ols_mse = np.ones((K))

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = X_train[train_index], X_test[test_index]
    y_train, y_test = y_train[train_index],y_test[test_index]
    [n_train, p_train], [n_test, p_test] = np.shape(X_train), np.shape(X_test)
    
    # include intercept
    off_train, off_test = np.ones(n_train), np.ones(n_test)
    M_train, M_test = np.c_[off_train, X_train], np.c_[off_test, X_test] # Include offset / intercept


# Ridge regression with k-fold cross validation

n, p = X_train.shape

k = 100; # try k values of lambda
lambdas = np.logspace(-4, 3, k)

# Number of folds
K = 10   

betas = np.zeros((k, p, k)) # all variable estimates
MSE = np.zeros((K, k))

# Actual manual failing cross-validation
'''

N = len(X_train)

I = np.asarray([0] * N)

for i in range(N):
    I[i] = (i + 1) % K + 1

I = I[np.random.permutation(N)]

for i in range(1, K+1):
    # Data permutation
    #X_validation_data = X_test[ i == I, : ]
    y_validation_data = Y_test[ i == I ]
    x_training_data = X_train[ i != I, : ]
    y_training_data = Y_train[ i != I ] 
'''
kf = KFold(10)
#kf.get_n_splits(X_train)

for train_index, test_index in kf.split(X_train):
    X_training_data, X_validation_data = X_train[train_index], X_test[test_index]
    y_training_data, y_validation_data = y_train[train_index], y_test[test_index]
    
    for j in range(0, len(lambdas)):
        # Make the ridge model
        ridge = Ridge(alpha=lambdas[j], fit_intercept=False)
        
        # Fit ridge
        ridge.fit(X_training_data, y_training_data)
        beta = ridge.coef_
        betas[(i-1), : , j] = beta
        MSE[(i-1), j ] = np.mean((y_validation_data - np.matmul(X_validation_data, beta))**2)
        
 
meanMSE = np.mean(MSE, axis = 0)
jOpt = np.argsort(meanMSE)[0]

lambda_OP = lambdas[jOpt]