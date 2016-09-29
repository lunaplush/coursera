# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 23:42:02 2016

@author: Inspiron
"""
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import pandas as pd



mu = 1
sigma = 0.2
norm_value = sts.norm(loc = mu, scale = sigma)

a = norm_value.rvs(10)

x = np.linspace(0,4,100)
b =norm_value.cdf(x)
plt.subplot(4,1,1)
plt.plot(x, b, "*")
plt.ylabel('$1F(x)$')
plt.xlabel('$x$')

c =norm_value.pdf(x)
plt.subplot(4,1,2)
plt.plot(x, c, "--")
plt.ylabel('$2f(x)$')
plt.xlabel('$x$')


beg = 3
end = 7
uniform_value = sts.uniform(beg, end-beg)
x = np.linspace(2,8,100)
b2 = uniform_value.cdf(x)
plt.subplot(4,1,3)
plt.plot(x,b2,"-")
plt.ylabel('$3F(x)$')
plt.xlabel('$x$')

c2 = uniform_value.pdf(x)
plt.subplot(4,1,4)
plt.plot(x, c2,"-")
plt.ylabel('$4F(x)$')
plt.xlabel('$x$')
