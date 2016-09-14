

import numpy as np

x = np.arange(1,15,0.1)
def func(x):
    return  np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)

import matplotlib.pyplot as plt

plt.plot(x,func(x))


import scipy as sc

a = np.array([[1,1],[1,15]])
b = np.array([func(1),func(15)])

X = sc.linalg.solve(a,b)

res = np.dot(a,X)

def funcA(x):
    return X[0]+X[1]*x

plt.plot(x,func(x),"-",x,funcA(x),"--")

#----------- 1,8,15
a = np.array([[1,1,1],[1,8,8*8],[1,15,15*15]])
b = np.array([func(1),func(8),func(15)])

X2 = sc.linalg.solve(a,b)

res = np.dot(a,X)

def funcA2(x):
    return X2[0]+X2[1]*x+X2[2]*x*x

plt.plot(x,func(x),"-",x,funcA2(x),"--")
