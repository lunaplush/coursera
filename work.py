# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:55:50 2017

@author: Inspiron
"""

import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
X = data.drop("Grant.Status", 1)
y = data["Grant.Status"]

def calculate_means(numeric_data):
    means = np.zeros(numeric_data.shape[1])
    for j in range(numeric_data.shape[1]):
        to_sum = numeric_data.iloc[:,j]
        indices = np.nonzero(~numeric_data.iloc[:,j].isnull())[0]
        if indices.size == 0: print("!!!",j) 
        correction = np.amax(to_sum[indices])
        to_sum /= correction
        for i in indices:
            means[j] += to_sum[i]
        means[j] /= indices.size
        means[j] *= correction
    return pd.Series(means, numeric_data.columns)
    
 
numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3', 
                'RFCD.Percentage.4', 'RFCD.Percentage.5',
                'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                'SEO.Percentage.4', 'SEO.Percentage.5',
                'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']
categorical_cols = list(set(X.columns.values.tolist()) - set(numeric_cols))    


 
X_real_zeros = X[numeric_cols].fillna(0.0)

means = calculate_means(data[numeric_cols])
X_real_mean = X[numeric_cols].fillna(means)

X_cat = pd.DataFrame(dict(zip(categorical_cols,[])))
for col in categorical_cols:
    indeces = X[col].isnull()
    A = X[col]
    A[indeces] ='NA'
    X_cat[col] = A.values.astype(str)


from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction import DictVectorizer as DV

encoder = DV(sparse = False)
X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())
    

from sklearn.cross_validation import train_test_split

(X_train_real_zeros, 
 X_test_real_zeros, 
 y_train, y_test) = train_test_split(X_real_zeros, y, 
                                     test_size=0.3, 
                                     random_state=0)
(X_train_real_mean, 
 X_test_real_mean) = train_test_split(X_real_mean, 
                                      test_size=0.3, 
                                      random_state=0)
(X_train_cat_oh,
 X_test_cat_oh) = train_test_split(X_cat_oh, 
                                   test_size=0.3, 
                                   random_state=0)
X_zeros = np.hstack((X_train_real_zeros, X_train_cat_oh))
X_mean = np.hstack((X_train_real_mean, X_train_cat_oh))    
