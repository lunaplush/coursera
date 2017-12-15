# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:14:46 2017

@author: Luna
"""
#conda install -c conda-forge brewer2mpl 

#https://penandpants.com/2012/07/27/brewer2mpl/ 

# О TCP-H тесте
# www.osp.ru/os/2000/11/178309


import brewer2mpl
from brewer2mpl import qualitative

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
#%%


bmap = brewer2mpl.get_map('Paired', 'Qualitative', 5)

cmap = bmap.get_mpl_colormap(N=1000, gamma=2.0)
bmap.mpl_colormap

#%%

inp = 32000

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])


#%matplotlib inline
#fig = plt.subplot()
fig,[ax1,ax2] = plt.subplots(1,2, sharey)
#yerr= df.quantile([0.75], axis = 1).values - df.quantile([0.25], axis = 1).values.reshape(4)
stds= np.sqrt(df.var(axis = 1))
means = df.mean(axis = 1)
conf_intervals = stats.norm.interval(0.05,loc = means, scale = stds)
conf_interval = conf_intervals[1] - conf_intervals[0]

colors = ['w','w','w','w']
for i in np.arange(len(means)):
    if inp > conf_intervals[0][i] and inp < conf_intervals[1][i] :
        colors[i] = 'w'
    if inp < conf_intervals[0][i] :
        colors[i] = 'b'
    if inp > conf_intervals[1][i] :    
        colors[i] = 'r'
        
ax1.bar(df.index, df.mean(axis = 1), yerr = conf_interval , color = colors, align  = "center")

ax1.set_xticks(df.index)
x_vals = ax1.get_xticks()
ax1.set_xticklabels(['{0:d}'.format(x) for x in x_vals])
ax1.plot([1991, 1996], [inp, inp],linestyle = 'dashed', color = "k")
ax2.boxplot(df.transpose()[1992],whis ='range')
