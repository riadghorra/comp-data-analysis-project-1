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

def RMSE (y, yhat):
    '''Calculates the relative mean squared error based of
    the predicted y-values and the correct ones.'''
    return np.sqrt(np.mean((y - yhat)**2)) / np.sqrt(np.mean((y - np.mean(y))**2))


# load and display data
data = pd.read_csv('Case1_Data.csv')
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

# split data in training / prediction based of Y value
training = data.loc[data['Y'].notnull()]
prediction = data.loc[data['Y'].isnull()]

print ('We have', len(training), 'training observations and', 
       len(prediction), 'prediction observations')

train_y = training['Y'].values
train_X = training[[col for col in training.columns if col != 'Y']].values


# One out of K encoding for categorical value
train_X = oneOutOfK(train_X, 99).astype(float)

# Flag for what handling of missing values to use
KNN_rpl = False


# replace missing values with mean of column
train_X_mn = missing_predictor_as_mean(train_X)

# replace missing values with mean of KNN
clean_data = missing_predictor_as_value(train_X, 0)
train_X_knn = missing_predictor_as_knn(5, train_X, clean_data)

# split Train / Test (70% / 30%) (for train_X)
test_split_ratio = 0.3

if (KNN_rpl):
    X_train, X_test, y_train , y_test= train_test_split(train_X_knn, train_y, test_size=test_split_ratio, random_state=0)
else:
    X_train, X_test, y_train , y_test= train_test_split(train_X_mn, train_y, test_size=test_split_ratio, random_state=0)


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


print('RMSE for OLS with {} replacement of NaN ='.format('knn' if KNN_rpl else 'mean'),  RMSE(y_test, yhat))

K = 2

# K - fold cross validation 
kf = KFold(K, shuffle=True)

ols_mse = np.ones((K))

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    # use _train_cv for the created subset of train and _val for the cv test
    # print("TRAIN CV:", len(train_index), "VAL:", len(test_index))
    X_train_cv, X_val = X_train[train_index], X_train[test_index]
    y_train_cv, y_val = y_train[train_index], y_train[test_index]
    [n_train_cv, p_train_cv], [n_val, p_val] = np.shape(X_train_cv), np.shape(X_val)
    
    # include intercept
    off_train_cv, off_val = np.ones(n_train_cv), np.ones(n_val)
    M_train_cv, M_val = np.c_[off_train_cv, X_train_cv], np.c_[off_val, X_val] # Include offset / intercept


# Ridge regression with k-fold cross validation

n, p = X_train.shape

k = 100; # try k values of lambda
lambdas = np.linspace(0, 100, k)

# Number of folds
K = 10   

betas = np.zeros((K, p, k)) # all variable estimates
training_error = np.zeros((K, k))
testing_error =  np.zeros((K, k))
MSE = np.zeros((K, k))
RMSE_ridge = np.zeros((K, k))

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

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    # use _train_cv for the created subset of train and _val for the cv test
    # print("TRAIN CV:", len(train_index), "VAL:", len(test_index))
    X_train_cv, X_val = X_train[train_index], X_train[test_index]
    y_train_cv, y_val = y_train[train_index], y_train[test_index]
    
    for j in range(0, len(lambdas)):
        # Make the ridge model
        ridge = Ridge(alpha=lambdas[j], fit_intercept=False)
        
        # Fit ridge
        ridge.fit(X_train_cv, y_train_cv)
        beta = ridge.coef_
        betas[(i-1), : , j] = beta
        y_hat_val = np.matmul(X_val, beta)
        y_hat_train = np.matmul(X_train, beta)

        MSE[(i-1), j ] = np.mean((y_val - y_hat_val)**2)
        RMSE_ridge[(i-1), j ] = RMSE(y_val, y_hat_val)
        training_error[(i-1), j ] = np.mean((y_train - y_hat_train)**2)
        testing_error[(i-1), j ] = np.mean((y_val - y_hat_val)**2)
 


mean_test_error = np.mean(testing_error, axis=0)
mean_training_error = np.mean(training_error, axis=0)

meanMSE = np.mean(MSE, axis = 0)
jOpt = np.argsort(meanMSE)[0]
lambda_OP = lambdas[jOpt]

meanRMSE = np.mean(RMSE_ridge, axis = 0)
jOpt_rmse = np.argsort(meanRMSE)[0]


lambda_OP = lambdas[jOpt]
lambda_OP_RMSE = lambdas[jOpt_rmse]

print('The optimal RMSE for ridge is with lambda =', lambda_OP_RMSE, 'and has RMSE of', np.min(RMSE_ridge))
plt.plot(lambdas, mean_training_error, color="green")
plt.plot(lambdas, mean_test_error, color="blue")
plt.show()

# average betas over 10 folds
mean_betas = np.mean(betas, axis=0)

plt.figure()
plt.semilogx(lambdas, mean_betas.T )
plt.xlabel("Lambdas")
plt.ylabel("Betas")
plt.title("Regularized beta estimates")

