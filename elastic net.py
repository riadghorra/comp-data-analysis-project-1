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



data = pd.read_csv('Case1_Data.csv')

training = data.loc[data['Y'].notnull()]
prediction = data.loc[data['Y'].isnull()]
train_y = training['Y'].as_matrix()
train_X = training[[col for col in training.columns if col != 'Y']].as_matrix()



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