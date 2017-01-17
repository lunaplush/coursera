

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



def funcA2(x):
    return X2[0]+X2[1]*x+X2[2]*x*x

plt.plot(x,func(x),"-",x,funcA2(x),"--")
#----------- 1,4,10,15
a = np.array([[1,1,1,1],[1,4,4*4,4*4*4],[1,10,10*10,10*10*10],[1,15,15*15,15*15*15]])
b = np.array([func(1),func(4),func(10),func(15)])

X3 = sc.linalg.solve(a,b)



def funcA3(x):
    return X3[0]+X3[1]*x+X3[2]*x*x+X3[3]*x*x*x

plt.plot(x,func(x),"-",x,funcA3(x),"--")
f = open("submission-2.txt", "w")
for i in range(4):
    f.write(str(round(X3[i],2)))
    f.write(" ")

f.close()