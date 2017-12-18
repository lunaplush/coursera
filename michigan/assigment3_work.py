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

inp = 47000


N = 3650

sigmas = np.array([200000.0,100000.0,140000.0,70000.0])
np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,sigmas[0],N), 
                   np.random.normal(43000,sigmas[1],N), 
                   np.random.normal(43500,sigmas[2],N), 
                   np.random.normal(48000,sigmas[3], N)], 
                  index=[1992,1993,1994,1995])


#%matplotlib inline
#fig = plt.subplot()
#fig,[ax1,ax2] = plt.subplots(1,2, sharey = False)
fig,ax1 = plt.subplots(1,1, sharey = False)
#onlinestatbook.com/2/estimation/mean.html
 
cmap=plt.get_cmap("seismic")

means = df.mean(axis = 1).values
sigma_1_96_M = 1.96*sigmas/np.sqrt(N)                
conf_intervals_mean_min =  means - sigma_1_96_M
conf_intervals_mean_max =  means + sigma_1_96_M                  
conf_interval_mean = (conf_intervals_mean_max - conf_intervals_mean_min)
                
colors = [cmap(0.5),cmap(0.5),cmap(0.5),cmap(0.5)]
for i in np.arange(len(means)):
    if inp >= conf_intervals_mean_min[i] and inp <= conf_intervals_mean_max[i] :
        colors[i] = cmap(0.5)
    #if inp <= conf_intervals_mean_min[i] - conf_interval_mean[i] :
    #    colors[i] = cmap(0.99)
   
    if inp >= conf_intervals_mean_max[i] + conf_interval_mean[i]:    
        colors[i] = cmap(0.99)
    
    if inp > conf_intervals_mean_max[i] and inp < conf_intervals_mean_max[i]+ conf_interval_mean[i]:    
        colors[i] = cmap(0.5*(inp - conf_intervals_mean_max[i])/conf_interval_mean[i]+0.5)
        print(colors[i])
        
    if inp < conf_intervals_mean_min[i] and inp > conf_intervals_mean_min[i] - conf_interval_mean[i] :
        colors[i] = cmap(0.5 - np.abs(inp -conf_intervals_mean_min[i])/(conf_interval_mean[i])/2)
    if inp < conf_intervals_mean_min[i] - conf_interval_mean[i] :
        colors[i] = cmap(0)
        
    
        
ax1.bar(df.index, df.mean(axis = 1), yerr = (conf_interval_mean/2).reshape(4) , color = colors, align  = "center", tick_label = df.index, ecolor = "k")
#ax1.scatter(1992,40000,color = cmap(0.55), linewidths  = 10)
ax1.set_xticks(df.index)
ax1.plot([1991, 1996], [inp, inp],linestyle = 'dashed', color = "k")



 #%%
