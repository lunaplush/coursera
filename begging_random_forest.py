# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:22:27 2017

@author: Inspiron
"""

from sklearn import datasets, cross_validation, tree, ensemble

def write_answer(answer,file):
    with open(file,'w') as f_out:
        f_out.write(str(answer))
        
data = datasets.load_digits()
X = data.data
y = data.target

#----1----
clf = tree.DecisionTreeClassifier(random_state = 1)
acc_cv = cross_validation.cross_val_score(clf,X,y, cv = 10)
acc= acc_cv.mean()
write_answer(acc, "answer_begging_1.txt")


#----2----
clf2 = ensemble.BaggingClassifier(n_estimators = 100)
acc_cv = cross_validation.cross_val_score(clf2,X,y,cv = 10)
acc2 = acc_cv.mean()
write_answer(acc2, "answer_begging_2.txt")