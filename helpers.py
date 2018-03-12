#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:24:37 2018

@author: wisse
"""

import numpy as np

# Helper functions for data handling

def RMSE (y, yhat):
    '''Calculates the relative mean squared error based of
    the predicted y-values and the correct ones.'''
    return np.sqrt(np.mean((y - yhat)**2)) / np.sqrt(np.mean((y - np.mean(y))**2))

def center(X):
    """ Center the columns (variables) of a data matrix to zero mean.
        
        X, MU = center(X) centers the observations of a data matrix such that each variable
        (column) has zero mean and also returns a vector MU of mean values for each variable.
     """ 
    n = X.shape[0]
    mu = np.mean(X,0)
    X = X - mu
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