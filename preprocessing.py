# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:32:23 2018

@author: Mark, Wisse, and Riad
"""
import numpy as np
import pandas as pd
from scipy.spatial import distance

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
    return data.astype(float)


def missing_predictor_as_mean(training_data):
    ''' Takes a data set in the form of a ndarray.
    It then replaces all NAN values with their 
    column mean'''
    data = np.copy(training_data)
    for i, column in enumerate(data.T):
            # column = train_X[column].as_matrix()
            column_without_nan = column[~np.isnan(np.array(column, dtype=float))]
            column_mean = column_without_nan.mean()
            # print( column_without_nan.mean() )
            for predictor_index in range(len(column)):
                if pd.isnull(np.array(column[predictor_index], dtype=float)):
                    column[predictor_index] = column_mean

            data.T[i, :] = column
    return data

def missing_predictor_as_value(data_set, value):
    ''' Takes a data set in the form of a ndarray,
    and a value. It then replaces all NAN values 
    with the given value'''
    data = np.copy(data_set)
    data[pd.isnull(np.array(data_set, dtype=float))] = value
    return data


def missing_predictor_as_knn(K, training_data, clean_data):
    ''' Takes a data set in the form of a ndarray, an 
    integer value k and a clean data set. It then 
    replaces all NAN values in the training data with
    a value computed with KNN. The value for k defines
    the number of nearest neighbours. The clean data set
    is used to do the actual distance calculation.'''
    data = np.copy(training_data)
    clean_data = clean_data
    n, p = training_data.shape
    # row major
    for i, row in enumerate(data):
        distances = np.zeros(n)
        for j in range(p):
            if pd.isnull(np.array(row[j], dtype=float)):
                # get the index of the nan
                # if the value is nan then we want to calculate the knn
                for k in range(n):
                    distances[k] = distance.euclidean(clean_data[i, :], clean_data[k, :])
                    
                # Calculate the estimated value of the NaN
                index = np.argsort(distances)[1:(K+1)]
                wt = sum(distances[index])
                W = distances[index] / wt
                
                nearest_neighbours = clean_data[index, j].astype(float)
                value = np.matmul(W, nearest_neighbours)
                data[i,j] = value

    return data