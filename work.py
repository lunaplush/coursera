# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:55:50 2017

@author: Inspiron
"""
#%%
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')

#%%

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

#%%
X_real_zeros = X[numeric_cols].fillna(0.0)

means = calculate_means(X[numeric_cols])
X_real_mean = X[numeric_cols].fillna(means)
#X_cat =  X[categorical_cols]
#for col in categorical_cols:
#    indeces = X_cat[col].isnull()
#    X_cat[col].set_value(indeces,"NA")
values = []
def iftrue(a,b):
    if a: return b
    
for col in categorical_cols:
    indeces = X[col].isnull()
 
    A = X[col]
    #A[indeces] ='NA'
    
    indeces_i = map(iftrue, indeces, range(len(indeces)))
    print(indeces_i)
    A.set_value(indeces_i,'NA')

    
    values.append(A.values.astype(str))

X_cat = pd.DataFrame.from_records(dict(zip(categorical_cols,values)))
#%%
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
X_zeros_train = np.hstack((X_train_real_zeros, X_train_cat_oh))
X_mean_train = np.hstack((X_train_real_mean, X_train_cat_oh))    

X_zeros_test = np.hstack((X_test_real_zeros, X_test_cat_oh))
X_mean_test = np.hstack((X_test_real_mean, X_test_cat_oh))   

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score

def plot_scores(optimizer):
    scores = [[item[0]['C'], 
               item[1], 
               (np.sum((item[2]-item[1])**2)/(item[2].size-1))**0.5] for item in optimizer.grid_scores_]
    scores = np.array(scores)
    plt.semilogx(scores[:,0], scores[:,1])
    plt.fill_between(scores[:,0], scores[:,1]-scores[:,2], 
                                  scores[:,1]+scores[:,2], alpha=0.3)
    plt.show()
    
def write_answer_1(auc_1, auc_2):
    auc = (auc_1 + auc_2)/2
    with open("preprocessing_lr_answer1.txt", "w") as fout:
        fout.write(str(auc))
    return(str(auc))    
        
param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
cv = 3

estimator = LogisticRegression(penalty = 'l2')
#print(estimator.get_params().keys())
optimizer = GridSearchCV(estimator, param_grid,cv = cv)
optimizer.fit(X_zeros_train, y_train)
plot_scores(optimizer)

estimator2 = LogisticRegression(penalty = 'l2')
optimizer2 = GridSearchCV(estimator2, param_grid, cv = cv)
optimizer2.fit(X_mean_train, y_train)
plot_scores(optimizer2)



y_zeros_predict = optimizer.best_estimator_.predict_proba(X_zeros_test)
auc_zeros = roc_auc_score(y_test, y_zeros_predict[:,1])

y_mean_predict = optimizer.best_estimator_.predict_proba(X_mean_test)
auc_mean = roc_auc_score(y_test,y_mean_predict[:,1])
print(write_answer_1(auc_mean,auc_zeros))

#%%
from sklearn.preprocessing import StandardScaler 
coder = StandardScaler()
coder.fit(X_zeros_train)
X_train_real_scaled = coder.transform(X_zeros_train)
X_work_test = coder.transform(X_zeros_test)