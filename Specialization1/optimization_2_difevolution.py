# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:24:05 2016

@author: Luna
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

import math


x = np.arange(1,30,0.1)
def func(x):
    return  np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
    
bounds = [(1,30)]


res = differential_evolution(func,bounds)


print(res)





plt.plot(x, func(x),"-",res.x[0],res.fun,"*")

f = open("submission-2-2.txt","w")
f.write(str(round(res.fun[0],2)))

f.close()