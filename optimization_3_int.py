# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:24:05 2016

@author: Luna
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

import math


x = np.arange(1,30,0.1)
def func(x):
    return  np.int_(np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2))
    
x0 = np.array([30])

res = minimize(func,x0,method="BFGS")

bounds =[(1,30)]
res2 = differential_evolution(func,bounds)


#print(res)





plt.plot(x,func(x),"-",res.x[0],res.fun,"*",res2.x[0],res2.fun,"*")


f = open("submission-2-3.txt","w")
f.write(str(round(res.fun[0],2)))
f.write(" ")
f.write(str(round(res2.fun,2)))

f.close()