#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:35:03 2018

@author: wisse
"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
train_X = oneOutOfK(train_X, 99)

print(train_y.dtype, train_X.dtype)


# replace missing values with mean of column
train_X_mn = missing_predictor_as_mean(train_X)

# replace missing values with mean of KNN
clean_data = missing_predictor_as_value(train_X, 0)
train_X_knn = missing_predictor_as_knn(5, train_X, clean_data)

# split Train / Test (70% / 30%) (for train_X with mean replacement)
X_train, X_test, y_train , y_test= train_test_split(train_X_mn, train_y, test_size=0.3, random_state=0)

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

print('MSE for OLS with mean replacement =', np.mean(res))

K = 2

# K - fold cross validation 
kf = KFold(K, shuffle=True)

ols_mse = np.ones((K))

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    [n_train, p_train], [n_test, p_test] = np.shape(X_train), np.shape(X_test)
    
    # include intercept
    off_train, off_test = np.ones(n_train), np.ones(n_test)
    M_train, M_test = np.c_[off_train, X_train], np.c_[off_test, X_test] # Include offset / intercept
    

   
    


