# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:51:24 2017

@author: Inspiron
"""

# -*- coding: utf-8 -*-
import numpy as np
from scipy import special
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

# -------------------
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score


def lossf(w, X, y, l1, l2):
    """
    Вычисление функции потерь.

    :param w: numpy.array размера  (M,) dtype = np.float
    :param X: numpy.array размера  (N, M), dtype = np.float
    :param y: numpy.array размера  (N,), dtype = np.int
    :param l1: float, l1 коэффициент регуляризатора 
    :param l2: float, l2 коэффициент регуляризатора 
    :return: float, значение функции потерь
    """
    print("-------w:",w[0:4])
    hlp = y*X.dot(w)
    print("-------losf",hlp[0:4])
    try:
        #lossf = np.sum(np.log2(1+np.exp(hlp))) + l1*np.sum(np.absolute(w)) + l2 * np.sum(np.power(w,2))
        #lossf = np.sum(-np.log2(special.expit(hlp))) + l1*np.sum(np.absolute(w)) + l2 * np.sum(np.power(w,2))
        lossf = np.sum(np.log1p((np.expm1(hlp)+1))/np.log(2)) + l1*np.sum(np.absolute(w)) + l2 * np.sum(np.power(w,2))
    
    except Exception as e:
        print(e)
        assert False, "1 Создание модели завершается с ошибкой"
   
    return lossf
    
def gradf(w, X, y, l1, l2):
    """
    Вычисление градиента функции потерь.

    :param w: numpy.array размера  (M,), dtype = np.float
    :param X: numpy.array размера  (N, M), dtype = np.float
    :param y: numpy.array размера  (N,), dtype = np.int
    :param l1: float, l1 коэффициент регуляризатора 
    :param l2: float, l2 коэффициент регуляризатора 
    :return: numpy.array размера  (M,), dtype = np.float, градиент функции потерь d lossf / dw
    """
   
    hlp = -y*X.dot(w)
    print("--**----grad",hlp[0:4])
   
    try:
        
        gradw=((np.expm1(hlp)+1)*special.expit(-hlp)*(-y)/np.log(2)).dot(X)+l1*np.sign(w)+2*l2*w
        #gradw=(special.expit(-hlp)*(-y)).dot(X)+l1*np.sign(w)+2*l2*w
    
    except Exception:
        assert False, "2 Error in grad" + Exception

    # Вам необходимо вычислить градиент функции потерь тут, решение может занимать 1 строку
    
    return gradw

class LR(ClassifierMixin, BaseEstimator):
    def __init__(self, lr=1, l1=1e-4, l2=1e-4, num_iter=1000, verbose=0):
        """
        Создание класса для лог регрессии
        
        :param lr: float, длина шага для оптимизатора
        :param l1: float, l1 коэффициент регуляризатора 
        :param l2: float, l2 коэффициент регуляризатора
        :param num_iter: int, число итераций оптимизатора
        :param verbose: bool, ключик для вывода
        """
        self.l1 = l1
        self.l2 = l2
        self.w = None
        self.lr = lr
        self.verbose = verbose
        self.num_iter = num_iter
    def fit(self, X, y):
        """
        Обучение логистической регрессии.
        Настраивает self.w коэффициенты модели.

        Если self.verbose == True, то выводите значение 
        функции потерь на итерациях метода оптимизации. 

        :param X: numpy.array размера  (N, M), dtype = np.float
        :param y: numpy.array размера  (N,), dtype = np.int
        :return: self
        """
       
        try: 
            n, d = X.shape            
            self.w = np.random.random_sample(d) - 0.5# Задайте начальное приближение вектора весов
            #self.w = np.zeros(d)
            num_iter_current = 0
            lossf_current = lossf(self.w, X,y,self.l1, self.l2)
            print ("Loss function on {0} iteration is {1} ".format(num_iter_current, lossf_current))
            
            num_iter_current = 1
            while np.absolute(lossf_current) > 1e-4  and num_iter_current < self.num_iter :
            #while  num_iter_current < self.num_iter :    
                #print("---------------------", num_iter_current)
                
                deltaL = gradf(self.w,X,y,self.l1, self.l2)
                print("       Gradient: ", deltaL[0:4])
                print("        self.w do ch:", self.w[0:4]  )
                self.w = self.w - self.lr * deltaL/n ##check +-!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
                lossf_current = lossf(self.w, X,y,self.l1, self.l2)
#                if(lossf_current == np.inf):
#                    self.w = self.w / self.w[0]
#                    lossf_current = lossf(self.w, X,y,self.l1, self.l2)
#                    print("           pri inf  w:",self.w[0:4])
                if self.verbose:
                    print("              w:",self.w[0:4])
                    print ("Loss function on {0} iteration is {1} ".format(num_iter_current, lossf_current))
                    
                num_iter_current = num_iter_current + 1
        except Exception as e:
            print("Error", e)
                
        #self.w = # Настройте параметры функции потерь с помощью градиентного спуска
        
        
        return self
        
print ("Start test")
X, y = make_classification(n_features=100, n_samples=1000, random_state = 0)
y = 2 * (y - 0.5)

(X_train,X_test,y_train,y_test) = train_test_split(X,y, test_size = 0.3, random_state = 0)
estimator = LogisticRegression(penalty = "l2")
estimator.fit(X_train, y_train)
y_predict =  estimator.predict_proba(X_test)
auc = roc_auc_score(y_test,y_predict[:,1])

#n, d = X.shape
#w =  np.random.random_sample(d)
#hlp = -y*X.dot(w)
#try:
#   deltaW=(np.exp(hlp)*special.expit(hlp)*(-y)).dot(X)+l1*np.sign(w)+2*l2*w
#except Exception:
#   assert False, "33 Error in grad" 
#w = w - deltaW
#hlp = -y*X.dot(w)
#A2 = np.sum(np.log(1+np.exp(hlp))) + l1*np.sum(np.absolute(w)) + l2 * np.sum(np.power(w,2))

try:
   clf = LR(lr=1, l1=1e-4, l2=1e-4, num_iter=1000, verbose=1)
   
except Exception:
   assert False, "Создание модели завершается с ошибкой"
   
try:
   clf = clf.fit(X, y)
except Exception as e:   
   assert False, "Обучение модели завершается с ошибкой"
print(clf.get_params())
