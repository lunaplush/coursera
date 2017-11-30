# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:12:14 2017

@author: Inspiron
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#%%
#%matplotlib 
n = 1000
fig, [[ax1, ax2],[ax3,ax4]] = plt.subplots(2,2)
fig.set_facecolor("w")

x1 = np.random.randn(n)
x2 = np.random.standard_gamma(1,size = n)
x3 = np.random.lognormal(mean = 0.0, sigma = 1.0, size = n )
x4 = np.random.normal(loc = 0.0, scale = 10, size = n)
k = 30
def update1(curr):

    if curr == k:
        a.event_source.stop()
   
    
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    i = (100 + 30*curr)
    bins1 = np.arange(-4, 4, 0.5)
    ax1.hist(x1[: i], bins = bins1 )  
    #ax2.ylim(0,300)
    ax2.annotate('n  = {}'.format(i),xycoords='figure fraction', xy = (0.4,0.95))
    bins2 = np.arange(0,8, 0.5)
    ax2.hist(x2[: i], bins = bins2 )
    bins3 = np.arange(0,50,5)
    ax3.hist(x3[: i], bins = bins3 )
    bins4 = np.arange(-40,40,5)
    ax4.hist(x4[: i], bins=bins4 )
    
a = animation.FuncAnimation(fig,update1,interval = 100)    

#%%


n = 100
x = np.random.randn(n)

# create the function that will do the plotting, where curr is the current frame
def update(curr):
    # check if animation is at the last frame, and if so, stop the animation a
    if curr == n: 
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4, 4, 0.5)
    plt.hist(x[:curr], bins=bins)
    plt.axis([-4,4,0,30])
    plt.gca().set_title('Sampling the Normal Distribution')
    plt.gca().set_ylabel('Frequency')
    plt.gca().set_xlabel('Value')
    plt.annotate('n = {}'.format(curr), [3,27])
    
fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval=100)    